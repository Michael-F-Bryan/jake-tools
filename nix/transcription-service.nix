# The native transcription service (transcription-service/), built with
# nixpkgs' Swift toolchain and Apple SDK. SwiftPM dependencies are vendored
# from Package.resolved by a fixed-output fetch; nothing is resolved at build
# or run time.
{
  lib,
  stdenv,
  swift,
  swiftpm,
  fetchSwiftPMDeps,
  apple-sdk_26,
  darwinMinVersionHook,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "transcription-service";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../transcription-service;
    fileset = lib.fileset.unions [
      ../transcription-service/Package.swift
      ../transcription-service/Package.resolved
      ../transcription-service/Sources
      # SwiftPM resolves every declared target, tests included.
      ../transcription-service/Tests
    ];
  };

  nativeBuildInputs = [
    swift
    swiftpm
  ];

  buildInputs = [
    apple-sdk_26
    # Package.swift's platform floor (FluidAudio's diarizer, Mutex).
    (darwinMinVersionHook "15.0")
  ];

  swiftpmDeps = fetchSwiftPMDeps {
    inherit (finalAttrs) pname version src;
    hash = "sha256-1Lqgap8stvYsS9X4+W5d4qYmGdD/neu7iAYrB5OLRX8=";
  };

  # SwiftPM fetches every binary target in the graph, even one only a
  # disabled trait links. FluidAudio's NeMo text-normalisation xcframework is
  # such a target (`traits: []` in Package.swift), so drop its declaration
  # from a writable copy of FluidAudio rather than download it.
  postPatch = ''
    vendored="$(readlink Packages)"
    rm Packages
    mkdir Packages
    ln -s "$vendored"/* Packages/
    rm Packages/FluidAudio
    cp -R "$vendored/FluidAudio" Packages/FluidAudio
    chmod -R u+w Packages/FluidAudio
    patch -p1 -d Packages/FluidAudio -i ${./fluidaudio-drop-nemo-binary-target.patch}
  '';

  # The Swift tests (mechanics, with a stub engine) run under this
  # toolchain via the flake's `checks` output, not on every package build.
  doCheck = false;
  # swiftpmHook's check phase, without XCTest: it ships only with Xcode, and
  # the tests use swift-testing.
  checkPhase = ''
    runHook preCheck
    local flagsArray=(-j "$NIX_BUILD_CORES" -c release --disable-xctest)
    concatTo flagsArray swiftpmFlags swiftpmFlagsArray
    TERM=dumb swift test "''${flagsArray[@]}"
    runHook postCheck
  '';

  meta = {
    description = "Local speech recognition and speaker diarisation over HTTP";
    mainProgram = "transcription-service";
    platforms = [ "aarch64-darwin" ];
  };
})
