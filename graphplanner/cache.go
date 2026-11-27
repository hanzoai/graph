/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// cachedNeighbor stores a single reachable authorized neighbor in a cached
// anchor neighborhood. All slices are deep-copied on Put and Get.
type cachedNeighbor struct {
	NodeID      uint32
	HopDistance int
	GraphScore  float64
	Path        []uint32
}

func cloneCachedNeighbors(in []cachedNeighbor) []cachedNeighbor {
	if in == nil {
		return nil
	}
	out := make([]cachedNeighbor, len(in))
	for i, n := range in {
		out[i] = cachedNeighbor{
			NodeID:      n.NodeID,
			HopDistance: n.HopDistance,
			GraphScore:  n.GraphScore,
		}
		if n.Path != nil {
			out[i].Path = make([]uint32, len(n.Path))
			copy(out[i].Path, n.Path)
		}
	}
	return out
}

// neighborhoodCacheKey uniquely scopes cached neighborhoods by tenant,
// canonicalized ACL scope fingerprint, tenant mutation epoch, anchor intern ID,
// sorted edge-type filter, and hop depth.
// Including aclFingerprint ensures callers in the same tenant with different
// ACL scopes can never share cached neighborhood entries.
type neighborhoodCacheKey struct {
	TenantID       string
	Epoch          uint64
	ACLFingerprint string
	AnchorID       uint32
	EdgeTypesKey   string
	TargetLabelKey string
	MaxHops        int
}

type lruCacheEntry struct {
	key       neighborhoodCacheKey
	neighbors []cachedNeighbor
}

// neighborhoodCache is a concurrency-safe bounded LRU cache with per-tenant
// epoch invalidation on any node, edge, ACL, or index mutation.
type neighborhoodCache struct {
	mu          sync.Mutex
	capacity    int
	ll          *list.List
	entries     map[neighborhoodCacheKey]*list.Element
	tenantEpoch map[string]uint64
	hits        uint64
	misses      uint64
	evictions   uint64
}

func newNeighborhoodCache(capacity int) *neighborhoodCache {
	if capacity <= 0 {
		capacity = 256
	}
	return &neighborhoodCache{
		capacity:    capacity,
		ll:          list.New(),
		entries:     make(map[neighborhoodCacheKey]*list.Element, capacity),
		tenantEpoch: make(map[string]uint64),
	}
}

func canonicalScopeFingerprint(scopes []string) string {
	if len(scopes) == 0 {
		return "none"
	}
	dedup := make([]string, 0, len(scopes))
	seen := make(map[string]bool, len(scopes))
	for _, s := range scopes {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		dedup = append(dedup, s)
	}
	if len(dedup) == 0 {
		return "none"
	}
	sort.Strings(dedup)
	h := sha256.New()
	for _, s := range dedup {
		fmt.Fprintf(h, "%d:%s\x00", len(s), s)
	}
	return hex.EncodeToString(h.Sum(nil))[:24]
}

func canonicalStringListKey(items []string) string {
	if len(items) == 0 {
		return "*"
	}
	dedup := make([]string, 0, len(items))
	seen := make(map[string]bool, len(items))
	for _, s := range items {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		dedup = append(dedup, s)
	}
	sort.Strings(dedup)
	return strings.Join(dedup, "\x1f")
}

func (c *neighborhoodCache) currentEpochLocked(tenantID string) uint64 {
	return c.tenantEpoch[tenantID]
}

// InvalidateTenant bumps the tenant's epoch and purges all entries for that
// tenant so stale nodes, edges, or revoked ACLs can never be served.
func (c *neighborhoodCache) InvalidateTenant(tenantID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tenantEpoch[tenantID]++
	for k, elem := range c.entries {
		if k.TenantID == tenantID {
			c.ll.Remove(elem)
			delete(c.entries, k)
		}
	}
}

// Clear drops all cached entries across every tenant and advances epochs.
func (c *neighborhoodCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for t := range c.tenantEpoch {
		c.tenantEpoch[t]++
	}
	c.ll.Init()
	c.entries = make(map[neighborhoodCacheKey]*list.Element, c.capacity)
}

// PeekHot checks whether a valid entry exists without mutating LRU order or
// hit/miss counters (used by the cost-based planner before execution).
func (c *neighborhoodCache) PeekHot(tenantID, aclFP string, anchorID uint32, edgeKey, labelKey string, maxHops int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := neighborhoodCacheKey{
		TenantID:       tenantID,
		Epoch:          c.currentEpochLocked(tenantID),
		ACLFingerprint: aclFP,
		AnchorID:       anchorID,
		EdgeTypesKey:   edgeKey,
		TargetLabelKey: labelKey,
		MaxHops:        maxHops,
	}
	_, ok := c.entries[key]
	return ok
}

// Get retrieves a deep copy of the cached neighborhood for the current tenant
// epoch, updating LRU recency and hit/miss counters.
func (c *neighborhoodCache) Get(tenantID, aclFP string, anchorID uint32, edgeKey, labelKey string, maxHops int) ([]cachedNeighbor, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := neighborhoodCacheKey{
		TenantID:       tenantID,
		Epoch:          c.currentEpochLocked(tenantID),
		ACLFingerprint: aclFP,
		AnchorID:       anchorID,
		EdgeTypesKey:   edgeKey,
		TargetLabelKey: labelKey,
		MaxHops:        maxHops,
	}
	elem, ok := c.entries[key]
	if !ok {
		c.misses++
		return nil, false
	}
	c.hits++
	c.ll.MoveToFront(elem)
	entry := elem.Value.(*lruCacheEntry)
	return cloneCachedNeighbors(entry.neighbors), true
}

// Put stores a deep copy of neighbors in the bounded LRU cache under the
// tenant's current epoch, evicting the least-recently-used entry if full.
func (c *neighborhoodCache) Put(tenantID, aclFP string, anchorID uint32, edgeKey, labelKey string, maxHops int, neighbors []cachedNeighbor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := neighborhoodCacheKey{
		TenantID:       tenantID,
		Epoch:          c.currentEpochLocked(tenantID),
		ACLFingerprint: aclFP,
		AnchorID:       anchorID,
		EdgeTypesKey:   edgeKey,
		TargetLabelKey: labelKey,
		MaxHops:        maxHops,
	}
	cloned := cloneCachedNeighbors(neighbors)
	if elem, ok := c.entries[key]; ok {
		c.ll.MoveToFront(elem)
		elem.Value.(*lruCacheEntry).neighbors = cloned
		return
	}
	if c.ll.Len() >= c.capacity {
		oldest := c.ll.Back()
		if oldest != nil {
			oldEntry := oldest.Value.(*lruCacheEntry)
			delete(c.entries, oldEntry.key)
			c.ll.Remove(oldest)
			c.evictions++
		}
	}
	elem := c.ll.PushFront(&lruCacheEntry{
		key:       key,
		neighbors: cloned,
	})
	c.entries[key] = elem
}

func (c *neighborhoodCache) Stats() (entries, capacity int, hits, misses, evictions uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ll.Len(), c.capacity, c.hits, c.misses, c.evictions
}
