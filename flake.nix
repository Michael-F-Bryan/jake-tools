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
      # Build the package and native Hermes plugin into an installer-chosen
      # interpreter's environment (nixfiles installs it into Hermes):
      #   jake-tools.lib.mkHermesPlugin {
      #     inherit pkgs python;
      #     lockedDependencies = [ ... ];  # what that environment lacks
      #   }
      lib.mkHermesPlugin = import ./nix/hermes-plugin.nix;

      # The native transcription service, built with nixpkgs' Swift and Apple
      # SDK, and its pinned models as an immutable `--models-dir`.
      packages.aarch64-darwin =
        let
          pkgs = nixpkgs.legacyPackages.aarch64-darwin;
        in
        {
          transcription-service = pkgs.callPackage ./nix/transcription-service.nix { };
          transcription-models = pkgs.callPackage ./nix/transcription-models.nix { };
        };

      # The Swift tests (mechanics, stub engine) under the same toolchain:
      #   nix build .#checks.aarch64-darwin.transcription-service
      checks.aarch64-darwin.transcription-service =
        self.packages.aarch64-darwin.transcription-service.overrideAttrs
          { doCheck = true; };

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
