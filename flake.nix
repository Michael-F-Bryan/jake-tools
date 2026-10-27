{
  description = "Development environment for jake-tools";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib stdenv;
        in
        {
          default = pkgs.mkShellNoCC {
            packages = [
              # Python side: `uv sync` installs the project, ruff, pyright,
              # pytest and pre-commit from uv.lock. A Nix Python 3.14 is on
              # PATH so uv can use it instead of downloading an interpreter.
              pkgs.uv
              pkgs.python314
              pkgs.git
              # Formats the Swift sources under transcription-service/.
              pkgs.swift-format
              # Generates and converts audio fixtures for the transcription
              # tests (macOS also has `say` and `afconvert` built in).
              pkgs.sox
              pkgs.nixfmt-rfc-style
            ];

            env = {
              # uv must not fetch its own interpreter when the Nix one fits.
              UV_PYTHON_PREFERENCE = "system";
            };

            shellHook = lib.optionalString stdenv.hostPlatform.isDarwin ''
              # The Swift service builds with Apple's toolchain, not nixpkgs'
              # (its Swift has no usable Darwin build). Xcode or the Command
              # Line Tools provide `swift`, CoreML and AVFoundation.
              if ! xcrun --find swift >/dev/null 2>&1; then
                echo "warning: Apple Swift toolchain not found; run 'xcode-select --install' to build transcription-service/" >&2
              fi
            '';
          };
        }
      );
    };
}
