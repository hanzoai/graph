/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// BenchmarkQuery defines a single evaluation query with ground-truth evidence,
// expected bridge entity path, and canonical answer.
type BenchmarkQuery struct {
	ID                   string        `json:"id"`
	Category             string        `json:"category"`
	Query                string        `json:"query"`
	Access               AccessContext `json:"access"`
	GoldSourceIDs        []string      `json:"gold_source_ids"`
	ExpectedPathEntities []string      `json:"expected_path_entities,omitempty"`
	GoldAnswer           string        `json:"gold_answer"`
	AnswerEntity         string        `json:"answer_entity"`
	RequiredArtifacts    []string      `json:"required_artifacts,omitempty"`
	TargetPredicate      string        `json:"target_predicate,omitempty"`
}

// WorkloadBundle groups canonical SourceRecords and BenchmarkQueries for a suite.
type WorkloadBundle struct {
	Name    string           `json:"name"`
	Records []SourceRecord   `json:"records"`
	Queries []BenchmarkQuery `json:"queries"`
}

const defaultTenant = "tenant-bench"

// BuildStandardBenchmarkSuite generates the deterministic multi-domain corpus
// and query suite covering HotpotQA-style 2-hop, MuSiQue-style 3-4 hop,
// cross-repository engineering incidents (including the CUDA graph memory alias
// chain), and simple single-hop factual queries.
func BuildStandardBenchmarkSuite() WorkloadBundle {
	var records []SourceRecord
	var queries []BenchmarkQuery

	r1, q1 := buildHotpotQA2Hop()
	records = append(records, r1...)
	queries = append(queries, q1...)

	r2, q2 := buildMuSiQue34Hop()
	records = append(records, r2...)
	queries = append(queries, q2...)

	r3, q3 := buildCrossRepoIncidents()
	records = append(records, r3...)
	queries = append(queries, q3...)

	r4, q4 := buildSimpleSingleHop()
	records = append(records, r4...)
	queries = append(queries, q4...)

	return WorkloadBundle{
		Name:    "hip-1347-hipporag2-standard-suite",
		Records: records,
		Queries: queries,
	}
}

func buildHotpotQA2Hop() ([]SourceRecord, []BenchmarkQuery) {
	type bridgeCase struct {
		id       string
		subject  string
		rel1     string
		bridge   string
		rel2     string
		answer   string
		question string
		topic    string
		distWord string
	}

	cases := []bridgeCase{
		{"hp01", "Aurelia_Voss", "designed_architecture_for", "Helios_Grid_Engine", "headquartered_in", "Zurich", "In which city is the organization headquartered that runs the engine designed by Aurelia Voss?", "distributed compute", "grid"},
		{"hp02", "Kestrel_Compiler", "authored_by", "Nadia_Kowalski", "graduated_from", "ETH_Lausanne", "Which university did the author of the Kestrel Compiler graduate from?", "compilers", "compiler"},
		{"hp03", "Obsidian_Observatory", "operated_by", "Nordic_Astro_Consortium", "founded_in_year", "1974", "In what year was the consortium founded that operates the Obsidian Observatory?", "astronomy", "observatory"},
		{"hp04", "Vesper_Protocol", "standardized_under", "IETF_Working_Group_94", "chaired_by", "Marcus_Delgado", "Who chairs the working group that standardized the Vesper Protocol?", "networking", "protocol"},
		{"hp05", "Tundra_Microkernel", "powers_satellite", "Borealis_Orbiter_4", "launched_from", "Kourou_Spaceport", "From which launch site was the satellite powered by Tundra Microkernel launched?", "aerospace", "microkernel"},
		{"hp06", "Cobalt_Codec", "patented_by", "Sora_Takahashi", "born_in", "Sapporo", "Where was the inventor who patented the Cobalt Codec born?", "media codecs", "codec"},
		{"hp07", "Mariner_Spectrometer", "deployed_aboard", "RV_Polar_Crest", "home_port_is", "Tromso", "What is the home port of the research vessel carrying the Mariner Spectrometer?", "oceanography", "spectrometer"},
		{"hp08", "Banyan_Database", "derived_storage_from", "Cedar_Log_Store", "licensed_under", "Apache_2_0", "Under what software license is the storage engine that Banyan Database derives from released?", "databases", "storage"},
		{"hp09", "Lumen_Lithography", "supplies_optics_to", "Valence_Foundry", "located_in", "Eindhoven", "In which city is the semiconductor foundry supplied by Lumen Lithography located?", "semiconductors", "lithography"},
		{"hp10", "Cinder_Consensus", "verified_using", "Alloy_Model_Checker", "created_at", "MIT_CSAIL", "At which research lab was the model checker used to verify Cinder Consensus created?", "formal methods", "consensus"},
		{"hp11", "Saffron_Battery", "uses_electrolyte", "Lanthanum_Garnet_7", "synthesized_by", "Kyoto_Materials_Lab", "Which laboratory synthesized the electrolyte used inside the Saffron Battery?", "energy storage", "battery"},
		{"hp12", "Quarry_Graph_Format", "adopted_by", "OpenMesh_Initiative", "funded_by", "Alpine_Science_Trust", "Which foundation funds the initiative that adopted Quarry Graph Format?", "graph formats", "graph"},
		{"hp13", "Solstice_Hydro_Dam", "built_across", "Kestrel_River_Basin", "drains_into", "Baltic_Sea", "Into which sea does the river basin spanned by Solstice Hydro Dam drain?", "hydrology", "hydro"},
		{"hp14", "Prism_Shader_IR", "targets_gpu_family", "Vortex_NPU_900", "manufactured_on", "3nm_GAA_Node", "On which process node is the accelerator targeted by Prism Shader IR manufactured?", "gpu shaders", "shader"},
		{"hp15", "Alder_Cryostat", "cools_detector", "Cherenkov_Array_East", "elevation_meters", "4200_meters", "What is the elevation of the detector array cooled by the Alder Cryostat?", "particle physics", "cryostat"},
		{"hp16", "Falcon_RPC_Wire", "replaces_framing_in", "Atlas_Control_Plane", "maintained_by", "Core_Infra_Guild", "Which team maintains the control plane whose framing was replaced by Falcon RPC Wire?", "rpc systems", "rpc"},
		{"hp17", "Granite_Type_System", "embedded_inside", "Opal_Query_Lang", "first_released_in", "2021", "In what year was the query language that embeds Granite Type System first released?", "type systems", "query"},
		{"hp18", "Beacon_Time_Sync", "calibrated_against", "Cesium_Fountain_C3", "housed_at", "Braunschweig_Metrology", "Where is the atomic clock used to calibrate Beacon Time Sync housed?", "timekeeping", "sync"},
		{"hp19", "Zephyr_Wind_Tunnel", "tested_airfoil", "NACA_Mod_64B", "adopted_on", "Skylark_Glider_X", "On which aircraft was the airfoil tested in the Zephyr Wind Tunnel adopted?", "aerodynamics", "airfoil"},
		{"hp20", "Drift_Anomaly_Model", "trained_on_dataset", "Seismic_Corpus_V8", "curated_by", "Pacific_Geo_Institute", "Which institute curated the dataset used to train Drift Anomaly Model?", "geophysics", "seismic"},
		{"hp21", "Harbor_Container_OS", "isolates_workloads_with", "Seccomp_Guard_V3", "audited_by", "Veritas_Kernel_Audit", "Which security firm audited the isolation layer used by Harbor Container OS?", "containers", "container"},
		{"hp22", "Orchid_Protein_Fold", "benchmarked_against", "CASP_Reference_15", "hosted_by", "UC_Davis_Genome_Center", "Which center hosted the reference benchmark used to evaluate Orchid Protein Fold?", "bioinformatics", "protein"},
		{"hp23", "Vector_Tile_Codec", "consumed_by", "Terra_Navigation_SDK", "written_in", "Go_1_26", "In which programming language is the navigation SDK that consumes Vector Tile Codec written?", "geospatial", "tile"},
		{"hp24", "Aegis_HSM_Module", "certified_under", "FIPS_140_3_Level_4", "accredited_by", "NIST_CMVP_Board", "Which accreditation board oversees the standard certifying Aegis HSM Module?", "cryptography", "hsm"},
	}

	var recs []SourceRecord
	var qs []BenchmarkQuery
	baseTS := int64(1730000000)

	for i, c := range cases {
		h1ID := c.id + "_hop1"
		h2ID := c.id + "_hop2"
		distID := c.id + "_dist"

		subDisp := strings.ReplaceAll(c.subject, "_", " ")
		brDisp := strings.ReplaceAll(c.bridge, "_", " ")
		ansDisp := strings.ReplaceAll(c.answer, "_", " ")

		// Hop 1: mentions subject -> bridge (does NOT contain the final answer).
		recs = append(recs, SourceRecord{
			SourceID:    h1ID,
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       subDisp,
			Text:        fmt.Sprintf("%s is a notable %s artifact. Primary link: %s %s %s.", subDisp, c.topic, subDisp, strings.ReplaceAll(c.rel1, "_", " "), brDisp),
			Timestamp:   baseTS + int64(i*10),
			Reliability: 0.94,
			Triples: []ExtractedTriple{
				{Subject: c.subject, Predicate: c.rel1, Object: c.bridge, Relevance: 0.95, Reliability: 0.94},
			},
		})

		// Hop 2: mentions bridge -> answer (intentionally omits subject name so lexical/dense retrieval misses it!).
		recs = append(recs, SourceRecord{
			SourceID:    h2ID,
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       brDisp,
			Text:        fmt.Sprintf("Reference profile for %s. Canonical fact: %s %s %s.", brDisp, brDisp, strings.ReplaceAll(c.rel2, "_", " "), ansDisp),
			Timestamp:   baseTS + int64(i*10+1),
			Reliability: 0.96,
			Triples: []ExtractedTriple{
				{Subject: c.bridge, Predicate: c.rel2, Object: c.answer, Relevance: 0.96, Reliability: 0.96},
			},
		})

		// Lexical distractor: repeats query keywords & subject name with irrelevant commentary and lower reliability.
		recs = append(recs, SourceRecord{
			SourceID:    distID,
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("%s %s survey note", subDisp, c.distWord),
			Text:        fmt.Sprintf("General commentary about %s and %s in %s. Mentions headquarters, city, year, university, founder, and team surveys without resolving the primary bridge record.", subDisp, c.distWord, c.topic),
			Timestamp:   baseTS + int64(i*10+2),
			Reliability: 0.42,
			Triples: []ExtractedTriple{
				{Subject: c.subject, Predicate: "commented_in", Object: "general_survey_archive", Relevance: 0.35, Reliability: 0.42},
			},
		})

		qs = append(qs, BenchmarkQuery{
			ID:                   c.id,
			Category:             "hotpotqa_2hop",
			Query:                c.question,
			Access:               AccessContext{TenantID: defaultTenant},
			GoldSourceIDs:        []string{h1ID, h2ID},
			ExpectedPathEntities: []string{NormalizeEntity(c.subject), NormalizeEntity(c.bridge), NormalizeEntity(c.answer)},
			GoldAnswer:           ansDisp,
			AnswerEntity:         NormalizeEntity(c.answer),
			TargetPredicate:      c.rel2,
		})
	}
	return recs, qs
}

func buildMuSiQue34Hop() ([]SourceRecord, []BenchmarkQuery) {
	type chainCase struct {
		id       string
		e1       string
		p1       string
		e2       string
		p2       string
		e3       string
		p3       string
		e4       string
		p4       string // optional 4th hop
		e5       string // optional 5th entity (for 4-hop chains)
		question string
		domain   string
	}

	cases := []chainCase{
		// 12 3-hop chains
		{"mq01", "Hyperion_Scheduler", "delegates_to", "Quartz_Queue_Broker", "persists_state_in", "Basalt_KV_Engine", "checksum_uses", "CRC32C_Castagnoli", "", "", "Which checksum algorithm is used by the KV engine backing the broker delegated to by Hyperion Scheduler?", "scheduling"},
		{"mq02", "Valkyrie_Router", "authenticates_via", "Keystone_Token_Issuer", "rotates_keys_with", "Vault_Transit_HSM", "signing_curve", "Ed25519_Pure", "", "", "Which signing curve is used by the HSM that rotates keys for the token issuer authenticating Valkyrie Router?", "security"},
		{"mq03", "Pelican_ETL_Pipeline", "ingests_stream", "Delta_Kafka_Topic", "partitioned_by", "Murmur3_Shard_Key", "defined_in_spec", "RFC_Shard_19", "", "", "Which specification defines the shard key partitioning the stream ingested by Pelican ETL Pipeline?", "data"},
		{"mq04", "Solon_Index_Service", "builds_graph_with", "DiskANN_Vamana_Builder", "quantizes_vectors_via", "Product_Quantizer_PQ8", "distance_metric", "Dot_Product_IP", "", "", "What distance metric is used by the quantizer inside the graph builder invoked by Solon Index Service?", "vector"},
		{"mq05", "Meridian_Billing_API", "reads_rates_from", "Tariff_Catalog_Cache", "invalidated_by", "CDC_Debezium_Relay", "deployed_in_region", "eu_west_frankfurt", "", "", "In which region is the CDC relay deployed that invalidates the cache read by Meridian Billing API?", "fintech"},
		{"mq06", "Kodiak_Build_Farm", "executes_actions_on", "Firecracker_MicroVM", "mounts_rootfs_from", "OverlayFS_Chunk_Store", "compression_codec", "Zstd_Level_3", "", "", "Which compression codec is used by the chunk store backing the microVM rootfs in Kodiak Build Farm?", "builds"},
		{"mq07", "Orion_Telemetry_Agent", "exports_spans_to", "Tempo_Ingester_Pool", "compacts_blocks_into", "Parquet_Columnar_V2", "retention_window", "90_days_cold", "", "", "What is the retention window of the columnar blocks produced by the ingester receiving Orion Telemetry Agent spans?", "observability"},
		{"mq08", "Cypress_Feature_Store", "serves_online_from", "Dragonfly_Shard_Mesh", "replicates_over", "RDMA_RoCEv2_Fabric", "mtu_bytes", "4096_bytes", "", "", "What MTU size is configured on the fabric replicating the shard mesh used by Cypress Feature Store?", "mlops"},
		{"mq09", "Cascade_DNS_Resolver", "forwards_queries_to", "Unbound_Recursive_Tier", "validates_dnssec_with", "KSK_Root_Anchor_2026", "key_algorithm", "ECDSAP256SHA256", "", "", "Which key algorithm secures the root anchor used by the recursive tier behind Cascade DNS Resolver?", "dns"},
		{"mq10", "Raptor_Search_Gateway", "rewrites_queries_via", "Synapse_Intent_Parser", "loads_lexicon_from", "Trie_Snapshot_V12", "built_by_job", "nightly_lexicon_dag", "", "", "Which job builds the lexicon snapshot loaded by the intent parser inside Raptor Search Gateway?", "search"},
		{"mq11", "Tartan_CI_Runner", "fetches_artifacts_from", "Harbor_OCI_Registry", "verifies_digest_via", "Cosign_Rekor_Log", "transparency_root", "Sigstore_Root_V4", "", "", "Which transparency root anchors the log verifying artifacts fetched by Tartan CI Runner?", "supply_chain"},
		{"mq12", "Zephyrus_Audio_DSP", "filters_frames_with", "Fir_Polyphase_Bank", "accelerated_by", "NEON_FMA_Kernel", "target_arch", "ARMv9_SVE2", "", "", "Which CPU architecture is targeted by the kernel accelerating the filter bank inside Zephyrus Audio DSP?", "dsp"},
		// 12 4-hop chains
		{"mq13", "Nebula_Training_Loop", "schedules_kernels_on", "Triton_Warp_Runtime", "allocates_scratch_from", "Arena_Slab_Manager", "evicts_blocks_via", "Clock_Pro_Policy", "tuned_by_flag", "max_slab_pressure_80", "Which tuning flag governs the eviction policy of the slab manager used by the runtime under Nebula Training Loop?", "ml_systems"},
		{"mq14", "Argus_Fraud_Scorer", "queries_features_from", "Graph_Neighborhood_Cache", "hydrated_by", "Flink_Window_Job", "checkpoints_into", "S3_Express_Bucket", "encrypted_with_key", "KMS_Alias_Fraud_09", "Which KMS key encrypts the checkpoint bucket of the stream job hydrating Argus Fraud Scorer's cache?", "risk"},
		{"mq15", "Helix_DNA_Aligner", "seeds_kmers_using", "Minimizer_Hash_Table", "sorted_in_memory_by", "Radix_MSD_Pass", "spills_partitions_to", "NVMe_Scratch_RAID0", "filesystem_type", "XFS_DirectIO", "Which filesystem hosts the scratch volume receiving spilled partitions from the sorter inside Helix DNA Aligner?", "genomics"},
		{"mq16", "Vanguard_Order_Book", "matches_ticks_in", "Disruptor_Ring_Buffer", "journals_fills_to", "Aeron_IPC_Archive", "replicated_to_dr_by", "Solarflare_Multicast", "nic_driver_version", "sfc_5_14_rt", "Which NIC driver version powers the multicast link replicating journals from Vanguard Order Book?", "trading"},
		{"mq17", "Cobalt_Edge_CDN", "terminates_tls_with", "BoringSSL_QUIC_Stack", "selects_origin_via", "Maglev_Consistent_Hash", "health_checked_by", "Envoy_Active_Prober", "failure_threshold", "3_consecutive_503", "What failure threshold triggers the prober guarding the consistent hash ring behind Cobalt Edge CDN?", "edge"},
		{"mq18", "Borealis_Climate_Sim", "solves_advection_with", "Spectral_Element_Grid", "exchanges_halos_via", "UCX_MPI_Transport", "pinned_to_numa_by", "Hwloc_Topology_Map", "interconnect_gen", "InfiniBand_NDR400", "Which interconnect generation carries the halo transport pinned for the solver in Borealis Climate Sim?", "hpc"},
		{"mq19", "Sentinel_WAF_Proxy", "inspects_payloads_with", "Hyperscan_DFA_Engine", "compiles_rules_from", "OWASP_CRS_Bundle_4", "signed_in_ci_by", "Release_Bot_Key_77", "audit_ticket", "SEC_REVIEW_2026_14", "Which audit ticket tracks the signing key for the rule bundle compiled into Sentinel WAF Proxy?", "appsec"},
		{"mq20", "Atlas_Map_Renderer", "tessellates_polygons_in", "Ear_Clipping_Worker", "uploads_buffers_via", "WebGPU_Staging_Belt", "synchronized_with", "Timeline_Semaphore_V2", "driver_workaround", "MESA_RADV_SYNC_FIX", "Which driver workaround guards the synchronization primitive used by the staging belt in Atlas Map Renderer?", "graphics"},
		{"mq21", "Chronos_Cron_Daemon", "acquires_lease_from", "Etcd_Raft_Cluster", "compacts_mvcc_via", "Auto_Compaction_Loop", "monitored_by_alert", "Mvcc_Db_Size_High", "runbook_id", "RB_ETCD_402", "Which runbook resolves the alert monitoring MVCC compaction on the cluster granting Chronos Cron Daemon leases?", "infra"},
		{"mq22", "Goliath_Video_Transcoder", "decodes_bitstream_with", "NVDEC_Hardware_Unit", "scales_frames_via", "NPP_Bicubic_Filter", "encodes_output_using", "AV1_SVT_Preset_6", "container_muxer", "MP4_Fragmented_CMAF", "Which container muxer packages output from the encoder downstream of Goliath Video Transcoder's hardware decoder?", "video"},
		{"mq23", "Hermes_Push_Gateway", "batches_tokens_in", "Priority_Leaky_Bucket", "dispatches_over", "HTTP2_Multiplex_Pool", "recovers_errors_with", "Decorrelated_Jitter_Backoff", "max_retry_cap", "12_seconds_cap", "What maximum retry cap bounds the backoff policy for the connection pool used by Hermes Push Gateway?", "messaging"},
		{"mq24", "Aether_LLM_Router", "estimates_load_from", "KV_Cache_Watermark", "routes_prefill_to", "Disaggregated_Prefill_Pod", "transfers_kv_over", "NIXL_RDMA_Channel", "fallback_transport", "TCP_ZeroCopy_Fallback", "Which fallback transport backs the KV transfer channel of the prefill pod selected by Aether LLM Router?", "inference"},
	}

	var recs []SourceRecord
	var qs []BenchmarkQuery
	baseTS := int64(1731000000)

	for i, c := range cases {
		var goldIDs []string
		var pathEnts []string

		addHop := func(suffix, subj, pred, obj string, hopIdx int) {
			sid := fmt.Sprintf("%s_%s", c.id, suffix)
			goldIDs = append(goldIDs, sid)
			sDisp := strings.ReplaceAll(subj, "_", " ")
			oDisp := strings.ReplaceAll(obj, "_", " ")
			pDisp := strings.ReplaceAll(pred, "_", " ")
			recs = append(recs, SourceRecord{
				SourceID:    sid,
				Version:     1,
				Kind:        SourceDocument,
				TenantID:    defaultTenant,
				Title:       sDisp,
				Text:        fmt.Sprintf("Technical specification for %s (%s). Link %d: %s %s %s.", sDisp, c.domain, hopIdx, sDisp, pDisp, oDisp),
				Timestamp:   baseTS + int64(i*20+hopIdx),
				Reliability: 0.95,
				Triples: []ExtractedTriple{
					{Subject: subj, Predicate: pred, Object: obj, Relevance: 0.96, Reliability: 0.95},
				},
			})
		}

		addHop("h1", c.e1, c.p1, c.e2, 1)
		addHop("h2", c.e2, c.p2, c.e3, 2)
		addHop("h3", c.e3, c.p3, c.e4, 3)

		finalAns := c.e4
		finalPred := c.p3
		pathEnts = []string{NormalizeEntity(c.e1), NormalizeEntity(c.e2), NormalizeEntity(c.e3), NormalizeEntity(c.e4)}
		if c.p4 != "" && c.e5 != "" {
			addHop("h4", c.e4, c.p4, c.e5, 4)
			finalAns = c.e5
			finalPred = c.p4
			pathEnts = append(pathEnts, NormalizeEntity(c.e5))
		}

		// Two lexical/dense distractors per chain that echo the query's opening entity and domain words.
		e1Disp := strings.ReplaceAll(c.e1, "_", " ")
		recs = append(recs, SourceRecord{
			SourceID:    c.id + "_dist1",
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("%s architecture overview", e1Disp),
			Text:        fmt.Sprintf("High-level roadmap note for %s in %s domain. Mentions checksums, signing curves, retention windows, tuning flags, runbooks, and fallback options in general terms.", e1Disp, c.domain),
			Timestamp:   baseTS + int64(i*20+10),
			Reliability: 0.38,
			Triples: []ExtractedTriple{
				{Subject: c.e1, Predicate: "overview_note", Object: "architecture_roadmap_digest", Relevance: 0.32, Reliability: 0.38},
			},
		})
		recs = append(recs, SourceRecord{
			SourceID:    c.id + "_dist2",
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("%s legacy migration log", e1Disp),
			Text:        fmt.Sprintf("Deprecated notes mentioning %s during %s quarterly review. Lists unrelated legacy defaults and superseded configurations.", e1Disp, c.domain),
			Timestamp:   baseTS + int64(i*20+11),
			Reliability: 0.32,
			Triples: []ExtractedTriple{
				{Subject: c.e1, Predicate: "legacy_note", Object: "deprecated_config_archive", Relevance: 0.28, Reliability: 0.32},
			},
		})

		qs = append(qs, BenchmarkQuery{
			ID:                   c.id,
			Category:             "musique_3_4hop",
			Query:                c.question,
			Access:               AccessContext{TenantID: defaultTenant},
			GoldSourceIDs:        goldIDs,
			ExpectedPathEntities: pathEnts,
			GoldAnswer:           strings.ReplaceAll(finalAns, "_", " "),
			AnswerEntity:         NormalizeEntity(finalAns),
			TargetPredicate:      finalPred,
		})
	}
	return recs, qs
}

func buildCrossRepoIncidents() ([]SourceRecord, []BenchmarkQuery) {
	type incidentSpec struct {
		id         string
		question   string
		failure    string
		kernel     string
		assumption string
		prID       string
		prTitle    string
		regTest    string
		component  string
	}

	specs := []incidentSpec{
		// Exact canonical scenario from hanzoai/graph#2!
		{
			id:         "inc01",
			question:   "Which change caused the CUDA graph memory alias?",
			failure:    "cuda_graph_memory_alias",
			kernel:     "stream_capture_kernel",
			assumption: "caching_allocator_pool",
			prID:       "pr_4821_allocator_reuse",
			prTitle:    "PR #4821: Aggressive cross-stream segment reuse in allocator pool",
			regTest:    "test_cuda_graph_alias_regression",
			component:  "gpu-runtime",
		},
		{
			id:         "inc02",
			question:   "Which change caused the WAL compaction checksum mismatch?",
			failure:    "wal_compaction_checksum_mismatch",
			kernel:     "segment_merge_iterator",
			assumption: "crc_frame_alignment_invariant",
			prID:       "pr_5104_zero_copy_wal_merge",
			prTitle:    "PR #5104: Zero-copy splice during segment merge",
			regTest:    "test_wal_merge_crc_regression",
			component:  "storage-wal",
		},
		{
			id:         "inc03",
			question:   "Which change caused the Raft snapshot lease split-brain?",
			failure:    "raft_snapshot_lease_split_brain",
			kernel:     "leader_read_index_handler",
			assumption: "monotonic_clock_lease_bound",
			prID:       "pr_3912_fast_lease_renewal",
			prTitle:    "PR #3912: Optimistic leader lease renewal before snapshot install",
			regTest:    "test_raft_snapshot_lease_regression",
			component:  "consensus",
		},
		{
			id:         "inc04",
			question:   "Which change caused the HNSW tombstone recall collapse?",
			failure:    "hnsw_tombstone_recall_collapse",
			kernel:     "ef_search_greedy_walker",
			assumption: "live_neighbor_bridge_invariant",
			prID:       "pr_6208_lazy_tombstone_prune",
			prTitle:    "PR #6208: Defer neighbor repair during batch vector deletes",
			regTest:    "test_hnsw_tombstone_bridge_regression",
			component:  "vector-index",
		},
		{
			id:         "inc05",
			question:   "Which change caused the AVX2 posting list decoder overrun?",
			failure:    "avx2_posting_decoder_overrun",
			kernel:     "groupvarint_simd_unpack",
			assumption: "tail_padding_32byte_guard",
			prID:       "pr_4419_compact_block_trim",
			prTitle:    "PR #4419: Trim trailing padding from packed UID blocks",
			regTest:    "test_avx2_unpack_tail_guard",
			component:  "posting-codec",
		},
		{
			id:         "inc06",
			question:   "Which change caused the gRPC keepalive RST flood?",
			failure:    "grpc_keepalive_rst_flood",
			kernel:     "http2_ping_frame_scheduler",
			assumption: "min_ping_interval_contract",
			prID:       "pr_5773_aggressive_liveness_probe",
			prTitle:    "PR #5773: Reduce client keepalive interval to 2s",
			regTest:    "test_http2_keepalive_enhance_your_calm",
			component:  "transport",
		},
		{
			id:         "inc07",
			question:   "Which change caused the memtable flush deadlock under pressure?",
			failure:    "memtable_flush_deadlock",
			kernel:     "immutable_table_rotator",
			assumption: "lock_order_write_then_flush",
			prID:       "pr_4980_inline_backpressure_stall",
			prTitle:    "PR #4980: Acquire write mutex inside flush callback",
			regTest:    "test_memtable_flush_lock_order",
			component:  "lsm-engine",
		},
		{
			id:         "inc08",
			question:   "Which change caused the epoll netpoller tail latency spike?",
			failure:    "epoll_netpoller_latency_spike",
			kernel:     "edge_triggered_socket_drainer",
			assumption: "bounded_batch_read_budget",
			prID:       "pr_6015_unbounded_socket_drain",
			prTitle:    "PR #6015: Drain socket until EAGAIN without yield",
			regTest:    "test_netpoller_fairness_budget",
			component:  "net-runtime",
		},
		{
			id:         "inc09",
			question:   "Which change caused the TLS session ticket decryption drop?",
			failure:    "tls_session_ticket_drop",
			kernel:     "edge_handshake_resumer",
			assumption: "dual_key_rotation_overlap",
			prID:       "pr_5341_single_slot_ticket_key",
			prTitle:    "PR #5341: Overwrite previous ticket key immediately on rotation",
			regTest:    "test_tls_ticket_overlap_window",
			component:  "edge-tls",
		},
		{
			id:         "inc10",
			question:   "Which change caused the fencing token wraparound write corruption?",
			failure:    "fencing_token_wraparound_corruption",
			kernel:     "distributed_lock_validator",
			assumption: "uint64_epoch_monotonicity",
			prID:       "pr_4127_uint32_wire_truncation",
			prTitle:    "PR #4127: Pack lock epoch into 32-bit wire header",
			regTest:    "test_fencing_token_64bit_roundtrip",
			component:  "lock-manager",
		},
		{
			id:         "inc11",
			question:   "Which change caused the KV cache prefix eviction storm?",
			failure:    "kv_cache_prefix_eviction_storm",
			kernel:     "radix_attention_block_table",
			assumption: "refcount_pinned_system_prompt",
			prID:       "pr_6892_lru_bypass_refcount",
			prTitle:    "PR #6892: Allow LRU sweeper to evict idle prefix pages",
			regTest:    "test_radix_prefix_pin_invariant",
			component:  "llm-serving",
		},
		{
			id:         "inc12",
			question:   "Which change caused the bitemporal as-of query phantom edge?",
			failure:    "bitemporal_phantom_edge_leak",
			kernel:     "valid_time_interval_scanner",
			assumption: "tx_time_upper_bound_filter",
			prID:       "pr_5509_skip_tx_filter_fastpath",
			prTitle:    "PR #5509: Bypass transaction-time filter on indexed range scans",
			regTest:    "test_bitemporal_as_of_phantom_guard",
			component:  "temporal-graph",
		},
		{
			id:         "inc13",
			question:   "Which change caused the Bloom filter false-negative key miss?",
			failure:    "bloom_filter_false_negative_miss",
			kernel:     "sstable_point_lookup_probe",
			assumption: "canonical_key_hash_seed",
			prID:       "pr_4672_per_shard_random_seed",
			prTitle:    "PR #4672: Randomize xxhash seed per SSTable builder",
			regTest:    "test_sstable_bloom_seed_consistency",
			component:  "sstable",
		},
		{
			id:         "inc14",
			question:   "Which change caused the TensorRT dynamic shape workspace OOM?",
			failure:    "tensorrt_dynamic_workspace_oom",
			kernel:     "execution_context_scratch_binder",
			assumption: "max_profile_workspace_reservation",
			prID:       "pr_6401_opt_profile_sizing",
			prTitle:    "PR #6401: Allocate scratch buffer from opt instead of max shape",
			regTest:    "test_tensorrt_max_shape_workspace",
			component:  "inference-trt",
		},
		{
			id:         "inc15",
			question:   "Which change caused the Prometheus histogram bucket NaN panic?",
			failure:    "prometheus_histogram_nan_panic",
			kernel:     "native_exponential_bucket_mapper",
			assumption: "positive_finite_observation_guard",
			prID:       "pr_5228_remove_isnan_fastcheck",
			prTitle:    "PR #5228: Drop NaN guard in hot latency observe path",
			regTest:    "test_histogram_nan_observation_safe",
			component:  "metrics",
		},
		{
			id:         "inc16",
			question:   "Which change caused the OAuth JWKS cache thundering herd?",
			failure:    "oauth_jwks_thundering_herd",
			kernel:     "jwt_keyset_refresh_client",
			assumption: "singleflight_refresh_coalescing",
			prID:       "pr_4755_per_request_kid_fetch",
			prTitle:    "PR #4755: Trigger uncoalesced JWKS fetch on unknown kid",
			regTest:    "test_jwks_singleflight_unknown_kid",
			component:  "auth-gateway",
		},
		{
			id:         "inc17",
			question:   "Which change caused the Parquet dictionary page truncation error?",
			failure:    "parquet_dict_page_truncation",
			kernel:     "rle_bitpacked_column_writer",
			assumption: "dict_fallback_threshold_check",
			prID:       "pr_5890_defer_plain_fallback",
			prTitle:    "PR #5890: Delay dictionary-to-plain fallback until row group end",
			regTest:    "test_parquet_dict_overflow_fallback",
			component:  "columnar-io",
		},
		{
			id:         "inc18",
			question:   "Which change caused the Kubernetes pod eviction PDB violation?",
			failure:    "k8s_eviction_pdb_violation",
			kernel:     "drain_controller_evictor",
			assumption: "disruption_budget_precheck_atomic",
			prID:       "pr_6112_parallel_node_drain_batch",
			prTitle:    "PR #6112: Issue concurrent eviction API calls per shard",
			regTest:    "test_drain_pdb_atomic_guard",
			component:  "cluster-ops",
		},
		{
			id:         "inc19",
			question:   "Which change caused the WebSocket backpressure memory leak?",
			failure:    "websocket_backpressure_memory_leak",
			kernel:     "outbound_frame_ring_flusher",
			assumption: "bounded_send_queue_drop_policy",
			prID:       "pr_4591_unbounded_slow_consumer_buf",
			prTitle:    "PR #4591: Grow outbound frame slice dynamically for slow peers",
			regTest:    "test_ws_slow_consumer_bounded_cap",
			component:  "realtime-ws",
		},
		{
			id:         "inc20",
			question:   "Which change caused the zero-copy ZAP schema version mismatch?",
			failure:    "zap_schema_version_mismatch",
			kernel:     "wire_frame_envelope_decoder",
			assumption: "capability_header_version_negotiation",
			prID:       "pr_6730_strip_minor_version_byte",
			prTitle:    "PR #6730: Compact envelope header by eliding schema minor byte",
			regTest:    "test_zap_envelope_version_compat",
			component:  "zap-wire",
		},
	}

	var recs []SourceRecord
	var qs []BenchmarkQuery
	baseTS := int64(1732000000)

	for i, s := range specs {
		evID := s.id + "_event"
		kernID := s.id + "_kernel"
		decID := s.id + "_decision"
		prSrcID := s.id + "_commit"
		testID := s.id + "_regtest"

		failDisp := strings.ReplaceAll(s.failure, "_", " ")
		kernDisp := strings.ReplaceAll(s.kernel, "_", " ")
		assumpDisp := strings.ReplaceAll(s.assumption, "_", " ")

		// 1. Agent failure event -> points to kernel behavior.
		recs = append(recs, SourceRecord{
			SourceID:    evID,
			Version:     1,
			Kind:        SourceAgentEvent,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Incident event: %s", failDisp),
			Text:        fmt.Sprintf("Production alert observed %s in component %s. Stack trace isolated execution inside %s.", failDisp, s.component, kernDisp),
			Timestamp:   baseTS + int64(i*30+1),
			Reliability: 0.94,
			Metadata:    map[string]string{"component": s.component},
			Triples: []ExtractedTriple{
				{Subject: s.failure, Predicate: "triggered_in", Object: s.kernel, Relevance: 0.96, Reliability: 0.94},
			},
		})

		// 2. Repository blob (kernel implementation) -> relies on allocator/system assumption.
		recs = append(recs, SourceRecord{
			SourceID:    kernID,
			Version:     1,
			Kind:        SourceRepoBlob,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Kernel file: %s", s.kernel),
			Text:        fmt.Sprintf("Source module %s (%s) depends strictly on %s for correctness under concurrency.", s.kernel, s.component, assumpDisp),
			Timestamp:   baseTS + int64(i*30+2),
			Reliability: 0.96,
			Metadata:    map[string]string{"component": s.component, "file": "pkg/" + s.component + "/" + s.kernel + ".go"},
			Triples: []ExtractedTriple{
				{Subject: s.kernel, Predicate: "assumes_invariant", Object: s.assumption, Relevance: 0.95, Reliability: 0.96},
			},
		})

		// 3. Architecture decision / invariant doc -> links assumption to PR change.
		recs = append(recs, SourceRecord{
			SourceID:    decID,
			Version:     1,
			Kind:        SourceDecision,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("ADR invariant: %s", s.assumption),
			Text:        fmt.Sprintf("Design invariant %s was altered by change %s, invalidating prior safety assumptions.", assumpDisp, s.prID),
			Timestamp:   baseTS + int64(i*30+3),
			Reliability: 0.95,
			Metadata:    map[string]string{"component": s.component},
			Triples: []ExtractedTriple{
				{Subject: s.assumption, Predicate: "broken_by", Object: s.prID, Relevance: 0.97, Reliability: 0.95},
			},
		})

		// 4. Git commit / PR record -> links PR to regression test (does NOT repeat failure keywords!).
		recs = append(recs, SourceRecord{
			SourceID:    prSrcID,
			Version:     1,
			Kind:        SourceGitCommit,
			TenantID:    defaultTenant,
			Title:       s.prTitle,
			Text:        fmt.Sprintf("Commit %s (%s) refactored %s behavior. Follow-up verification added in %s.", s.prID, s.prTitle, assumpDisp, s.regTest),
			Timestamp:   baseTS + int64(i*30+4),
			Reliability: 0.98,
			Metadata:    map[string]string{"component": s.component, "pr": s.prID},
			Triples: []ExtractedTriple{
				{Subject: s.prID, Predicate: "verified_by", Object: s.regTest, Relevance: 0.96, Reliability: 0.98},
			},
		})

		// 5. Regression test file -> verifies fix for PR.
		recs = append(recs, SourceRecord{
			SourceID:    testID,
			Version:     1,
			Kind:        SourceRepoBlob,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Regression test: %s", s.regTest),
			Text:        fmt.Sprintf("Automated test suite %s reproduces the violation introduced by %s and guards %s permanently.", s.regTest, s.prID, assumpDisp),
			Timestamp:   baseTS + int64(i*30+5),
			Reliability: 0.99,
			Metadata:    map[string]string{"component": s.component, "test": s.regTest},
			Triples: []ExtractedTriple{
				{Subject: s.regTest, Predicate: "guards_against", Object: s.prID, Relevance: 0.95, Reliability: 0.99},
			},
		})

		// 2 noisy keyword-heavy distractors per incident (low reliability chat/speculation logs).
		recs = append(recs, SourceRecord{
			SourceID:    s.id + "_noise1",
			Version:     1,
			Kind:        SourceAgentEvent,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Triage chat about %s", failDisp),
			Text:        fmt.Sprintf("Speculative discussion asking which change caused %s in %s. Someone guessed an unrelated driver upgrade before checking kernel or PR history.", failDisp, s.component),
			Timestamp:   baseTS + int64(i*30+12),
			Reliability: 0.30,
			Triples: []ExtractedTriple{
				{Subject: s.failure, Predicate: "rumored_in", Object: "triage_chat_backlog", Relevance: 0.30, Reliability: 0.30},
			},
		})
		recs = append(recs, SourceRecord{
			SourceID:    s.id + "_noise2",
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Old wiki FAQ: %s troubleshooting", failDisp),
			Text:        fmt.Sprintf("Outdated troubleshooting checklist for %s in %s suggesting node reboot and log rotation. Does not identify the culprit PR or regression test.", failDisp, s.component),
			Timestamp:   baseTS + int64(i*30+13),
			Reliability: 0.34,
			Triples: []ExtractedTriple{
				{Subject: s.failure, Predicate: "mentioned_in", Object: "legacy_faq_reboot_guide", Relevance: 0.28, Reliability: 0.34},
			},
		})

		qs = append(qs, BenchmarkQuery{
			ID:                   s.id,
			Category:             "cross_repo_incidents",
			Query:                s.question,
			Access:               AccessContext{TenantID: defaultTenant},
			GoldSourceIDs:        []string{evID, kernID, decID, prSrcID, testID},
			ExpectedPathEntities: []string{s.failure, s.kernel, s.assumption, s.prID, s.regTest},
			GoldAnswer:           s.prID,
			AnswerEntity:         s.prID,
			RequiredArtifacts:    []string{prSrcID, testID},
			TargetPredicate:      "broken_by",
		})
	}
	return recs, qs
}

func buildSimpleSingleHop() ([]SourceRecord, []BenchmarkQuery) {
	type factCase struct {
		id       string
		entity   string
		pred     string
		value    string
		question string
	}

	cases := []factCase{
		{"sh01", "zapdb_default_port", "configured_as", "port_9080", "What is the configured value for zapdb_default_port?"},
		{"sh02", "dql_parser_max_depth", "configured_as", "depth_64", "What is the configured value for dql_parser_max_depth?"},
		{"sh03", "posting_list_split_size", "configured_as", "size_256kb", "What is the configured value for posting_list_split_size?"},
		{"sh04", "ristretto_cache_ratio", "configured_as", "ratio_0_35", "What is the configured value for ristretto_cache_ratio?"},
		{"sh05", "hnsw_default_m_links", "configured_as", "links_16", "What is the configured value for hnsw_default_m_links?"},
		{"sh06", "hnsw_default_ef_construct", "configured_as", "ef_200", "What is the configured value for hnsw_default_ef_construct?"},
		{"sh07", "bloom_bits_per_key", "configured_as", "bits_10", "What is the configured value for bloom_bits_per_key?"},
		{"sh08", "wal_sync_interval_ms", "configured_as", "interval_50ms", "What is the configured value for wal_sync_interval_ms?"},
		{"sh09", "snapshot_chunk_bytes", "configured_as", "chunk_4mb", "What is the configured value for snapshot_chunk_bytes?"},
		{"sh10", "compaction_worker_threads", "configured_as", "threads_4", "What is the configured value for compaction_worker_threads?"},
		{"sh11", "uid_lease_batch_size", "configured_as", "batch_10000", "What is the configured value for uid_lease_batch_size?"},
		{"sh12", "schema_lock_timeout_sec", "configured_as", "timeout_30s", "What is the configured value for schema_lock_timeout_sec?"},
		{"sh13", "tokenizer_max_ngram_len", "configured_as", "ngram_8", "What is the configured value for tokenizer_max_ngram_len?"},
		{"sh14", "geohash_precision_bits", "configured_as", "precision_12", "What is the configured value for geohash_precision_bits?"},
		{"sh15", "metrics_scrape_path", "configured_as", "path_metrics_v1", "What is the configured value for metrics_scrape_path?"},
		{"sh16", "healthcheck_probe_path", "configured_as", "path_healthz", "What is the configured value for healthcheck_probe_path?"},
		{"sh17", "max_query_concurrency", "configured_as", "concurrency_512", "What is the configured value for max_query_concurrency?"},
		{"sh18", "tx_abort_retention_min", "configured_as", "retention_15m", "What is the configured value for tx_abort_retention_min?"},
		{"sh19", "raft_heartbeat_tick_ms", "configured_as", "tick_100ms", "What is the configured value for raft_heartbeat_tick_ms?"},
		{"sh20", "tls_min_protocol_ver", "configured_as", "tls_1_3", "What is the configured value for tls_min_protocol_ver?"},
	}

	var recs []SourceRecord
	var qs []BenchmarkQuery
	baseTS := int64(1733000000)

	for i, c := range cases {
		sid := c.id + "_doc"
		recs = append(recs, SourceRecord{
			SourceID:    sid,
			Version:     1,
			Kind:        SourceDocument,
			TenantID:    defaultTenant,
			Title:       c.entity,
			Text:        fmt.Sprintf("Configuration manual entry: %s is configured as %s in production defaults.", c.entity, c.value),
			Timestamp:   baseTS + int64(i),
			Reliability: 0.98,
			Triples: []ExtractedTriple{
				{Subject: c.entity, Predicate: c.pred, Object: c.value, Relevance: 0.98, Reliability: 0.98},
			},
		})
		qs = append(qs, BenchmarkQuery{
			ID:                   c.id,
			Category:             "simple_single_hop",
			Query:                c.question,
			Access:               AccessContext{TenantID: defaultTenant},
			GoldSourceIDs:        []string{sid},
			ExpectedPathEntities: []string{c.entity, c.value},
			GoldAnswer:           c.value,
			AnswerEntity:         c.value,
			TargetPredicate:      c.pred,
		})
	}
	return recs, qs
}

// BuildHubCollapseNegativeWorkload constructs the negative boundary regime
// `neg_unweighted_hub_collapse`: takes the cross-repo incidents and injects
// high-degree generic hub entities attached to 60 low-reliability noisy telemetry
// records so disabling reliability weighting, hub damping, and recognition memory
// causes severe hub-flood collapse.
func BuildHubCollapseNegativeWorkload() WorkloadBundle {
	recs, qs := buildCrossRepoIncidents()
	hubNames := []string{
		"prod_cluster_hub",
		"runtime_telemetry_hub",
		"memory_subsystem_hub",
		"kernel_alert_hub",
	}

	// Attach each incident's entry failure entity weakly to a shared generic hub
	// (simulating real telemetry where every incident also emits a generic tag).
	for i := range recs {
		if strings.HasSuffix(recs[i].SourceID, "_event") {
			hub := hubNames[i%len(hubNames)]
			if len(recs[i].Triples) > 0 {
				failEnt := recs[i].Triples[0].Subject
				recs[i].Triples = append(recs[i].Triples, ExtractedTriple{
					Subject:     failEnt,
					Predicate:   "emitted_on",
					Object:      hub,
					Relevance:   0.85,
					Reliability: 0.35,
				})
			}
		}
	}

	// Inject 60 noisy low-reliability telemetry logs linked to the hubs.
	baseTS := int64(1734000000)
	for i := 0; i < 60; i++ {
		hub := hubNames[i%len(hubNames)]
		noiseEnt := fmt.Sprintf("noisy_telemetry_burst_%02d", i)
		recs = append(recs, SourceRecord{
			SourceID:    fmt.Sprintf("hub_noise_log_%02d", i),
			Version:     1,
			Kind:        SourceAgentEvent,
			TenantID:    defaultTenant,
			Title:       fmt.Sprintf("Telemetry noise burst %02d on %s", i, hub),
			Text:        fmt.Sprintf("High-volume noisy telemetry %s emitted on %s reporting routine memory, kernel, cache, and allocator jitter.", noiseEnt, hub),
			Timestamp:   baseTS + int64(i),
			Reliability: 0.25,
			Triples: []ExtractedTriple{
				{Subject: hub, Predicate: "logged_burst", Object: noiseEnt, Relevance: 0.90, Reliability: 0.25},
				{Subject: noiseEnt, Predicate: "related_hub", Object: hubNames[(i+1)%len(hubNames)], Relevance: 0.85, Reliability: 0.25},
			},
		})
	}

	return WorkloadBundle{
		Name:    "neg_unweighted_hub_collapse",
		Records: recs,
		Queries: qs,
	}
}

// BuildACLBridgeNegativeWorkload constructs the negative boundary regime
// `neg_acl_bridge_leakage_control`: two tenants share identical entity names
// (e.g. cuda_graph_memory_alias, stream_capture_kernel, caching_allocator_pool)
// AND tenant-alpha contains restricted-ACL records sharing entity names with
// public records.
func BuildACLBridgeNegativeWorkload() ([]SourceRecord, []BenchmarkQuery) {
	sharedChains := [][3]string{
		{"cuda_graph_memory_alias", "stream_capture_kernel", "caching_allocator_pool"},
		{"wal_compaction_checksum_mismatch", "segment_merge_iterator", "crc_frame_alignment_invariant"},
		{"raft_snapshot_lease_split_brain", "leader_read_index_handler", "monotonic_clock_lease_bound"},
		{"hnsw_tombstone_recall_collapse", "ef_search_greedy_walker", "live_neighbor_bridge_invariant"},
		{"avx2_posting_decoder_overrun", "groupvarint_simd_unpack", "tail_padding_32byte_guard"},
		{"grpc_keepalive_rst_flood", "http2_ping_frame_scheduler", "min_ping_interval_contract"},
		{"memtable_flush_deadlock", "immutable_table_rotator", "lock_order_write_then_flush"},
		{"epoll_netpoller_latency_spike", "edge_triggered_socket_drainer", "bounded_batch_read_budget"},
		{"tls_session_ticket_drop", "edge_handshake_resumer", "dual_key_rotation_overlap"},
		{"fencing_token_wraparound_corruption", "distributed_lock_validator", "uint64_epoch_monotonicity"},
	}

	var recs []SourceRecord
	var qs []BenchmarkQuery
	baseTS := int64(1735000000)

	for i, ch := range sharedChains {
		e1, e2, e3 := ch[0], ch[1], ch[2]

		// Tenant-alpha PUBLIC record: mentions e1 -> e2.
		recs = append(recs, SourceRecord{
			SourceID:    fmt.Sprintf("alpha_pub_%02d", i),
			Version:     1,
			Kind:        SourceAgentEvent,
			TenantID:    "tenant-alpha",
			ACL:         []string{"team:public"},
			Title:       fmt.Sprintf("Alpha public incident %s", e1),
			Text:        fmt.Sprintf("Public alpha note: %s observed in %s.", e1, e2),
			Timestamp:   baseTS + int64(i*10),
			Reliability: 0.92,
			Triples: []ExtractedTriple{
				{Subject: e1, Predicate: "triggered_in", Object: e2, Relevance: 0.95, Reliability: 0.92},
			},
		})

		// Tenant-alpha RESTRICTED ACL record: shares e2 -> e3 -> alpha_restricted_fix.
		recs = append(recs, SourceRecord{
			SourceID:    fmt.Sprintf("alpha_restricted_%02d", i),
			Version:     1,
			Kind:        SourceDecision,
			TenantID:    "tenant-alpha",
			ACL:         []string{"sec:incident-response"},
			Title:       fmt.Sprintf("Alpha restricted advisory %s", e2),
			Text:        fmt.Sprintf("Confidential alpha security advisory: %s relies on %s and internal mitigation alpha_sec_patch_%02d.", e2, e3, i),
			Timestamp:   baseTS + int64(i*10+1),
			Reliability: 0.97,
			Triples: []ExtractedTriple{
				{Subject: e2, Predicate: "assumes_invariant", Object: e3, Relevance: 0.96, Reliability: 0.97},
				{Subject: e3, Predicate: "mitigated_by", Object: fmt.Sprintf("alpha_sec_patch_%02d", i), Relevance: 0.96, Reliability: 0.97},
			},
		})

		// Tenant-beta CONFIDENTIAL record: shares IDENTICAL entity names e1, e2, e3!
		recs = append(recs, SourceRecord{
			SourceID:    fmt.Sprintf("beta_conf_%02d", i),
			Version:     1,
			Kind:        SourceGitCommit,
			TenantID:    "tenant-beta",
			ACL:         []string{"tenant-beta:core"},
			Title:       fmt.Sprintf("Beta confidential patch for %s", e1),
			Text:        fmt.Sprintf("Confidential tenant-beta fix: %s via %s and %s resolved by beta_private_pr_%02d.", e1, e2, e3, i),
			Timestamp:   baseTS + int64(i*10+2),
			Reliability: 0.99,
			Triples: []ExtractedTriple{
				{Subject: e1, Predicate: "triggered_in", Object: e2, Relevance: 0.98, Reliability: 0.99},
				{Subject: e2, Predicate: "assumes_invariant", Object: e3, Relevance: 0.98, Reliability: 0.99},
				{Subject: e3, Predicate: "broken_by", Object: fmt.Sprintf("beta_private_pr_%02d", i), Relevance: 0.98, Reliability: 0.99},
			},
		})

		// Query 1: tenant-alpha public caller querying e1 (must NEVER see beta_conf or alpha_restricted).
		qs = append(qs, BenchmarkQuery{
			ID:            fmt.Sprintf("acl_bridge_alpha_pub_%02d", i),
			Category:      "acl_bridge_control",
			Query:         fmt.Sprintf("Which change caused %s in %s?", strings.ReplaceAll(e1, "_", " "), strings.ReplaceAll(e2, "_", " ")),
			Access:        AccessContext{TenantID: "tenant-alpha", Scopes: []string{"team:public"}},
			GoldSourceIDs: []string{fmt.Sprintf("alpha_pub_%02d", i)},
		})

		// Query 2: tenant-gamma (zero records) querying e1 (must return 0 hits, 0 foreign inspections).
		qs = append(qs, BenchmarkQuery{
			ID:       fmt.Sprintf("acl_bridge_foreign_tenant_%02d", i),
			Category: "acl_bridge_control",
			Query:    fmt.Sprintf("Which change caused %s in %s?", strings.ReplaceAll(e1, "_", " "), strings.ReplaceAll(e2, "_", " ")),
			Access:   AccessContext{TenantID: "tenant-gamma", Scopes: []string{"team:public"}},
		})
	}

	return recs, qs
}

// --- External Dataset Loader (Real HotpotQA JSON, MuSiQue JSONL, or HIP-1347 JSONL) ---

type hotpotQAItem struct {
	ID              string  `json:"_id"`
	Question        string  `json:"question"`
	Answer          string  `json:"answer"`
	SupportingFacts [][]any `json:"supporting_facts"`
	Context         [][]any `json:"context"`
}

type musiqueParagraph struct {
	Idx           int    `json:"idx"`
	Title         string `json:"title"`
	ParagraphText string `json:"paragraph_text"`
	IsSupporting  bool   `json:"is_supporting"`
}

type musiqueItem struct {
	ID         string             `json:"id"`
	Question   string             `json:"question"`
	Answer     string             `json:"answer"`
	Paragraphs []musiqueParagraph `json:"paragraphs"`
}

// LoadExternalDataset loads an external benchmark file in any of three formats:
//  1. Real HotpotQA JSON array (`.json`)
//  2. Real MuSiQue JSONL (`.jsonl`, one object per line with `paragraphs`)
//  3. Native HIP-1347 WorkloadBundle JSON (`{"name","records","queries"}`)
func LoadExternalDataset(path string) (WorkloadBundle, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return WorkloadBundle{}, fmt.Errorf("hipporag: read external dataset %s: %w", path, err)
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return WorkloadBundle{}, fmt.Errorf("hipporag: external dataset %s is empty", path)
	}

	// Format 1: Native WorkloadBundle JSON object with "records" and "queries".
	// Attempt whole-document native decoding before falling back to JSONL so
	// formatting/indentation (e.g., json.MarshalIndent(bundle, "", "")) does not
	// misclassify a native bundle as JSONL.
	if strings.HasPrefix(trimmed, "{") {
		var native WorkloadBundle
		if err := json.Unmarshal(raw, &native); err == nil && len(native.Records) > 0 && len(native.Queries) > 0 {
			if native.Name == "" {
				native.Name = filepath.Base(path)
			}
			return native, nil
		}
	}

	// Format 2: HotpotQA JSON array.
	if strings.HasPrefix(trimmed, "[") {
		var items []hotpotQAItem
		if err := json.Unmarshal(raw, &items); err != nil {
			return WorkloadBundle{}, fmt.Errorf("hipporag: parse HotpotQA JSON %s: %w", path, err)
		}
		return convertHotpotQA(filepath.Base(path), items), nil
	}

	// Format 3: MuSiQue JSONL (one JSON object per line).
	var mItems []musiqueItem
	scanner := bufio.NewScanner(strings.NewReader(trimmed))
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var item musiqueItem
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return WorkloadBundle{}, fmt.Errorf("hipporag: parse MuSiQue JSONL line in %s: %w", path, err)
		}
		mItems = append(mItems, item)
	}
	if err := scanner.Err(); err != nil {
		return WorkloadBundle{}, fmt.Errorf("hipporag: scan external dataset %s: %w", path, err)
	}
	if len(mItems) == 0 {
		return WorkloadBundle{}, fmt.Errorf("hipporag: no items parsed from %s", path)
	}
	return convertMuSiQue(filepath.Base(path), mItems), nil
}

func hotpotArticleSourceID(title, text string) string {
	h := sha256.New()
	fmt.Fprintf(h, "%d:%s\x00%d:%s", len(title), title, len(text), text)
	digest := hex.EncodeToString(h.Sum(nil))[:24]
	if normTitle := NormalizeEntity(title); normTitle != "" {
		return fmt.Sprintf("ext_hp_%s_%s", normTitle, digest)
	}
	return "ext_hp_" + digest
}

func convertHotpotQA(name string, items []hotpotQAItem) WorkloadBundle {
	seenDoc := make(map[string]bool)
	var recs []SourceRecord
	var qs []BenchmarkQuery

	for idx, it := range items {
		suppTitles := make(map[string]bool)
		for _, sf := range it.SupportingFacts {
			if len(sf) >= 1 {
				if title, ok := sf[0].(string); ok {
					suppTitles[title] = true
				}
			}
		}

		var goldIDs []string
		seenGold := make(map[string]bool)
		for _, ctxEntry := range it.Context {
			if len(ctxEntry) < 2 {
				continue
			}
			title, ok := ctxEntry[0].(string)
			if !ok {
				continue
			}
			sentsAny, ok := ctxEntry[1].([]any)
			if !ok {
				continue
			}
			var parts []string
			for _, sa := range sentsAny {
				if s, ok := sa.(string); ok {
					parts = append(parts, s)
				}
			}
			text := strings.Join(parts, " ")
			sid := hotpotArticleSourceID(title, text)
			if !seenDoc[sid] {
				seenDoc[sid] = true
				recs = append(recs, SourceRecord{
					SourceID:    sid,
					Version:     1,
					Kind:        SourceDocument,
					TenantID:    defaultTenant,
					Title:       title,
					Text:        text,
					Timestamp:   int64(1736000000 + len(recs)),
					Reliability: 0.92,
				})
			}
			if suppTitles[title] && !seenGold[sid] {
				seenGold[sid] = true
				goldIDs = append(goldIDs, sid)
			}
		}

		qid := it.ID
		if qid == "" {
			qid = fmt.Sprintf("ext_hp_q_%d", idx)
		}
		qs = append(qs, BenchmarkQuery{
			ID:            qid,
			Category:      "external_hotpotqa",
			Query:         it.Question,
			Access:        AccessContext{TenantID: defaultTenant},
			GoldSourceIDs: goldIDs,
			GoldAnswer:    it.Answer,
			AnswerEntity:  NormalizeEntity(it.Answer),
		})
	}
	return WorkloadBundle{Name: "external:" + name, Records: recs, Queries: qs}
}

func musiquePassageSourceID(title, text string) string {
	h := sha256.New()
	fmt.Fprintf(h, "%d:%s\x00%d:%s", len(title), title, len(text), text)
	digest := hex.EncodeToString(h.Sum(nil))[:24]
	if normTitle := NormalizeEntity(title); normTitle != "" {
		return fmt.Sprintf("ext_mq_%s_%s", normTitle, digest)
	}
	return "ext_mq_" + digest
}

func convertMuSiQue(name string, items []musiqueItem) WorkloadBundle {
	seenDoc := make(map[string]bool)
	var recs []SourceRecord
	var qs []BenchmarkQuery

	for idx, it := range items {
		var goldIDs []string
		seenGold := make(map[string]bool)
		for _, p := range it.Paragraphs {
			sid := musiquePassageSourceID(p.Title, p.ParagraphText)
			if !seenDoc[sid] {
				seenDoc[sid] = true
				recs = append(recs, SourceRecord{
					SourceID:    sid,
					Version:     1,
					Kind:        SourceDocument,
					TenantID:    defaultTenant,
					Title:       p.Title,
					Text:        p.ParagraphText,
					Timestamp:   int64(1736100000 + len(recs)),
					Reliability: 0.92,
				})
			}
			if p.IsSupporting && !seenGold[sid] {
				seenGold[sid] = true
				goldIDs = append(goldIDs, sid)
			}
		}

		qid := it.ID
		if qid == "" {
			qid = fmt.Sprintf("ext_mq_q_%d", idx)
		}
		qs = append(qs, BenchmarkQuery{
			ID:            qid,
			Category:      "external_musique",
			Query:         it.Question,
			Access:        AccessContext{TenantID: defaultTenant},
			GoldSourceIDs: goldIDs,
			GoldAnswer:    it.Answer,
			AnswerEntity:  NormalizeEntity(it.Answer),
		})
	}
	return WorkloadBundle{Name: "external:" + name, Records: recs, Queries: qs}
}
