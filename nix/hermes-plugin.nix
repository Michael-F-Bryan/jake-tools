# jake-tools as a Python package for an existing interpreter's environment,
# for installation into Hermes' sealed runtime (see nixfiles).
#
# The host environment already supplies most of jake-tools' dependencies and
# must not receive second copies (Hermes refuses colliding distributions), so
# none are propagated except `lockedDependencies`: the names the host lacks,
# installed from the exact wheels pinned in uv.lock for this platform.
# Nothing here checks the remaining dependencies are present; the installer
# verifies the assembled runtime imports the plugin.
{
  pkgs,
  python,
  lockedDependencies,
}:

let
  inherit (pkgs) lib;
  pythonPackages = python.pkgs;
  lock = builtins.fromTOML (builtins.readFile ../uv.lock);
  project = (builtins.fromTOML (builtins.readFile ../pyproject.toml)).project;

  # Wheel tags this interpreter can load: pure, or this CPython's own ABI,
  # on any platform or this one.
  cpython = "cp${lib.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
  platformMarkers =
    {
      aarch64-linux = [
        "manylinux"
        "_aarch64"
      ];
      x86_64-linux = [
        "manylinux"
        "_x86_64"
      ];
      aarch64-darwin = [
        "macosx"
        "_arm64"
      ];
    }
    .${pkgs.stdenv.hostPlatform.system};
  compatible =
    wheel:
    let
      tags = lib.splitString "-" (lib.removeSuffix ".whl" (baseNameOf wheel.url));
      count = builtins.length tags;
      pythonTag = builtins.elemAt tags (count - 3);
      abiTag = builtins.elemAt tags (count - 2);
      platformTag = builtins.elemAt tags (count - 1);
    in
    (pythonTag == "py3" && abiTag == "none" || pythonTag == cpython && abiTag == cpython)
    && (platformTag == "any" || lib.all (marker: lib.hasInfix marker platformTag) platformMarkers);

  locked =
    name:
    let
      entry = lib.findFirst (p: p.name == name) (throw "uv.lock does not lock ${name}") lock.package;
      wheel =
        lib.findFirst compatible (throw "uv.lock has no ${name} wheel for ${cpython} on this platform")
          entry.wheels;
    in
    pythonPackages.buildPythonPackage {
      pname = name;
      inherit (entry) version;
      format = "wheel";
      src = pkgs.fetchurl { inherit (wheel) url hash; };
      # Its own dependencies come from the host environment, like ours.
      dontCheckRuntimeDeps = true;
      doCheck = false;
    };
in
pythonPackages.buildPythonPackage {
  pname = project.name;
  inherit (project) version;
  pyproject = true;
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../pyproject.toml
      ../README.md
      # Stray bytecode in a working tree is not source.
      (lib.fileset.fileFilter (file: !(lib.hasSuffix ".pyc" file.name)) ../src)
    ];
  };
  build-system = [ pythonPackages.uv-build ];
  # pyproject pins the uv_build series uv itself uses for development; the
  # host's nixpkgs ships a newer one that builds the same wheel.
  pypaBuildFlags = [ "--skip-dependency-check" ];
  dependencies = map locked lockedDependencies;
  dontCheckRuntimeDeps = true;
  doCheck = false;
}
