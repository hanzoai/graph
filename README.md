# Hanzo Graph

Hanzo Graph forks Dgraph (`github.com/dgraph-io/dgraph`, Apache License 2.0) at commit 64804bd2 and
cuts it down to what runs inside a single process. It keeps the posting-list engine, the schema and
index layers, the tokenizers, the type system, the DQL parser, the UID set algebra, and the wire
types — 18 packages, 99 source files, 44 test files, 410 passing tests. Nothing here serves or
clusters: no Raft, no gRPC service, no query executor, no integration harness, and the only two
`main` packages are a build-variable emitter and a codec benchmark. Keys and posting lists live in
`github.com/luxfi/zapdb` v1.10.6, which the caller opens in managed mode and for which the caller
supplies every start and commit timestamp. DQL parses to a syntax tree and stops there — no package
imports the parser, and the executor that walked that tree belonged to the cluster code — so a
caller reads by key or by index token and composes UID lists with `algo` instead of traversing,
filtering, or binding variables.
