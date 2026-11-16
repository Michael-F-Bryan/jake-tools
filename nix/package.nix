# jake-tools as a standalone application: `bin/jake-tools` and
# `bin/jake-tools-mcp` on their own Python, with exactly the runtime
# dependencies pinned in uv.lock, installed from the locked wheels (uv2nix).
{
  pkgs,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
}:

let
  inherit (pkgs) lib stdenv;

  # Only what uv2nix and the build read, so unrelated edits don't rebuild.
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../uv.lock
      ../README.md
      # Stray bytecode in a working tree is not source.
      (lib.fileset.fileFilter (file: !(lib.hasSuffix ".pyc" file.name)) ../src)
    ];
  };

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      python = pkgs.python314;
    }).overrideScope
      (
        lib.composeManyExtensions [
          # Supplies uv_build, which builds jake-tools itself.
          pyproject-build-systems.overlays.wheel
          (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        ]
      );

  # Runtime dependencies only; the lockfile's dev group stays out.
  venv = pythonSet.mkVirtualEnv "jake-tools-env" workspace.deps.default;

  package = (pkgs.callPackage pyproject-nix.build.util { }).mkApplication {
    inherit venv;
    package = pythonSet.jake-tools;
  };
in
package.overrideAttrs (old: {
  passthru = (old.passthru or { }) // {
    tests.smoke = pkgs.runCommand "jake-tools-smoke" { nativeBuildInputs = [ package ]; } ''
      jake-tools --help > /dev/null
      jake-tools-mcp --help > /dev/null
      touch $out
    '';
  };
  meta = (old.meta or { }) // {
    mainProgram = "jake-tools-mcp";
  };
})
