#!/bin/sh
# `swift test` for transcription-service.
#
# Under the Command Line Tools (no Xcode) SwiftPM does not pass the Testing
# framework's macro plugin directory to the compiler, so `@Test` fails with
# "plugin for module 'TestingMacros' not found". Pass it explicitly when the
# active developer directory has one; Xcode toolchains need nothing extra.
set -eu
cd "$(dirname "$0")"
plugins="$(xcode-select -p)/usr/lib/swift/host/plugins/testing"
if [ -d "$plugins" ]; then
    set -- -Xswiftc -plugin-path -Xswiftc "$plugins" "$@"
fi
exec swift test "$@"
