#!/usr/bin/env bash
# Project-local cargo wrapper: the toolchain lives inside ./rustup and ./cargo,
# so every invocation must point RUSTUP_HOME / CARGO_HOME / PATH there.
export RUSTUP_HOME=/home/z/my-project/.rustup
export CARGO_HOME=/home/z/my-project/.cargo
export PATH=/home/z/my-project/.cargo/bin:$PATH
exec cargo "$@"
