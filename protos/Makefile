#
# SPDX-FileCopyrightText: © 2017-2026 Istari Digital, Inc.
# SPDX-License-Identifier: Apache-2.0
#

UNAME_S := $(shell uname -s)

DGO_PB_VERSION = 250

DGO_PATH   := github.com/dgraph-io/dgo/v${DGO_PB_VERSION}
TMPDIR     := $(shell mktemp -d)
PROTO_PATH := ${TMPDIR}/src:.
PROTO_PATH := ${PROTO_PATH}:${TMPDIR}/src/${DGO_PATH}/protos

.PHONY: help
help:
	@echo Use \"make regenerate\" to proceed.

.PHONY: clean
clean:
	@mkdir -p pb && rm -f pb/pb.pb.go

.PHONY: tidy-deps
tidy-deps:
	@go mod tidy -v

.PHONY: check
check:
	@./depcheck.sh && \
		(echo "Installing proto libraries to versions in go.mod." ; \
		go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31.0)

.PHONY: copy-protos
copy-protos:
	@mkdir -p ${TMPDIR}/src/${DGO_PATH}/protos
	@cp $(shell go list -m -f "{{.Dir}}" ${DGO_PATH})/protos/api.proto \
	 ${TMPDIR}/src/${DGO_PATH}/protos/api.proto

.PHONY: patch-pb
patch-pb:
	./patch_pb.sh

ifneq ($(UNAME_S),Linux)
.PHONY: regenerate
regenerate:
	@echo "Running in Docker..."
	@docker run --rm -v $(shell pwd)/..:/go/src/github.com/hanzoai/graph \
		-w /go/src/github.com/hanzoai/graph/protos \
		golang:1.25.0 \
		/bin/bash -c "apt-get update && apt-get install -y protobuf-compiler && make regenerate"
else
.PHONY: regenerate
regenerate: tidy-deps copy-protos check clean
	echo ${DGO_PATH}
	@protoc \
		--proto_path=/usr/local/include \
		--proto_path=/usr/include \
		--proto_path=/opt/homebrew/include/google/protobuf \
		--proto_path=${PROTO_PATH} \
		--go_out=pb \
		--go_opt=paths=source_relative \
		pb.proto
	@$(MAKE) patch-pb
	@rm -rf ${TMPDIR}
	@echo Done.
endif
