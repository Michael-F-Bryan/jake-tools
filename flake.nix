{
  description = "Development environment for jake-tools";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # The standalone Jake package on every system; on macOS also the native
      # transcription service, built with nixpkgs' Swift and Apple SDK, and its
      # pinned models as an immutable `--models-dir`.
      packages = forAllSystems (
        pkgs:
        let
          jake-tools = pkgs.callPackage ./nix/package.nix { python = pkgs.python314; };
        in
        {
          # `bin/jake-tools` and `bin/jake-tools-mcp` on their own Python with
          # the dependencies pinned in uv.lock; Hermes launches the latter as
          # a stdio MCP server (docs/hermes-mcp.md).
          inherit jake-tools;
          default = jake-tools;
        }
        // nixpkgs.lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "aarch64-darwin") {
          transcription-service = pkgs.callPackage ./nix/transcription-service.nix { };
          transcription-models = pkgs.callPackage ./nix/transcription-models.nix { };
        }
      );

      # `jake-tools` smoke test: both entry points start (the build itself
      # imports `jake_tools.mcp.server`).
      #
      # The Swift tests (mechanics, stub engine) run under the same toolchain:
      #   nix build .#checks.aarch64-darwin.transcription-service
      checks =
        nixpkgs.lib.recursiveUpdate
          (forAllSystems (pkgs: {
            jake-tools = self.packages.${pkgs.stdenv.hostPlatform.system}.jake-tools.tests.smoke;
          }))
          {
            aarch64-darwin.transcription-service =
              self.packages.aarch64-darwin.transcription-service.overrideAttrs
                { doCheck = true; };
          };

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = [
            # Python side: `uv sync` installs the project, ruff, pyright,
            # pytest and pre-commit from uv.lock. A Nix Python 3.14 is on
            # PATH so uv can use it instead of downloading an interpreter.
            pkgs.uv
            pkgs.python314
            pkgs.git
            # Formats the Swift sources under transcription-service/, which
            # build with `nix build .#transcription-service`.
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
        };
      });
    };
}
