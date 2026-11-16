# jake-tools as a standalone application: `bin/jake-tools` and
# `bin/jake-tools-mcp` on their own nixpkgs Python, with exactly the runtime
# dependency closure pinned in uv.lock, installed from the locked wheels.
#
# The closure starts at the `jake-tools` package's `dependencies` (never the
# dev group) and follows each package's `dependencies`, evaluating the
# markers uv.lock records for this platform. An unrecognised marker is an
# error, not a guess. Nothing is installed at run time and no checkout is
# needed.
{
  pkgs,
  python ? pkgs.python314,
}:

let
  inherit (pkgs) lib stdenv;
  pythonPackages = python.pkgs;
  lock = builtins.fromTOML (builtins.readFile ../uv.lock);
  project = (builtins.fromTOML (builtins.readFile ../pyproject.toml)).project;

  # Wheel tags this interpreter can load: pure, abi3 up to this CPython, or
  # this CPython's own ABI, on any platform or this one.
  minor = lib.toInt (lib.last (lib.splitString "." python.pythonVersion));
  cpython = "cp3${toString minor}";
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
    .${stdenv.hostPlatform.system};
  cpythonMinor =
    tag:
    let
      m = builtins.match "cp3([0-9]+)" tag;
    in
    if m == null then null else lib.toInt (builtins.head m);
  compatible =
    wheel:
    let
      tags = lib.splitString "-" (lib.removeSuffix ".whl" (baseNameOf wheel.url));
      count = builtins.length tags;
      pythonTags = lib.splitString "." (builtins.elemAt tags (count - 3));
      abiTag = builtins.elemAt tags (count - 2);
      platformTags = lib.splitString "." (builtins.elemAt tags (count - 1));
      pythonOk =
        (lib.elem "py3" pythonTags && abiTag == "none")
        || (lib.elem cpython pythonTags && (abiTag == cpython || abiTag == "none"))
        || (abiTag == "abi3" && lib.any (t: cpythonMinor t != null && cpythonMinor t <= minor) pythonTags);
      platformOk =
        lib.elem "any" platformTags
        || lib.any (tag: lib.all (marker: lib.hasInfix marker tag) platformMarkers) platformTags;
    in
    pythonOk && platformOk;

  # The environment markers evaluate against. Only variables that occur in
  # uv.lock are defined; anything else is rejected.
  environment = {
    sys_platform = if stdenv.hostPlatform.isDarwin then "darwin" else "linux";
    platform_python_implementation = "CPython";
    implementation_name = "cpython";
  };
  evalClause =
    clause:
    let
      m = builtins.match "[ ]*([a-z_]+)[ ]*(==|!=)[ ]*'([^']*)'[ ]*" clause;
      variable = builtins.elemAt m 0;
      operator = builtins.elemAt m 1;
      value = builtins.elemAt m 2;
    in
    if m == null then
      throw "uv.lock marker clause not understood: ${clause}"
    else if !(environment ? ${variable}) then
      throw "uv.lock marker variable not supported: ${variable} (in ${clause})"
    else if operator == "==" then
      environment.${variable} == value
    else
      environment.${variable} != value;
  markerHolds = marker: lib.all evalClause (lib.splitString " and " marker);

  entryFor =
    name: lib.findFirst (p: p.name == name) (throw "uv.lock does not lock ${name}") lock.package;

  # Names reachable from the project's own dependencies, by work list.
  # A dependency on `name[extra]` also pulls in the packages that extra
  # lists in `[package.optional-dependencies]`. Nodes are `name` or
  # `name[extra]`.
  closure =
    let
      step =
        state:
        if state.pending == [ ] then
          state
        else
          let
            node = builtins.head state.pending;
            rest = builtins.tail state.pending;
          in
          if lib.elem node state.seen then
            step (state // { pending = rest; })
          else
            step {
              seen = state.seen ++ [ node ];
              pending = rest ++ (nodeDependencies node);
            };
      seen =
        (step {
          seen = [ ];
          pending = liveDependencies (entryFor project.name);
        }).seen;
    in
    lib.unique (map (node: (parseNode node).name) seen);
  parseNode =
    node:
    let
      m = builtins.match "([^]\\[]+)[[](.+)[]]" node;
    in
    if m == null then
      {
        name = node;
        extra = null;
      }
    else
      {
        name = builtins.elemAt m 0;
        extra = builtins.elemAt m 1;
      };
  nodeDependencies =
    node:
    let
      parsed = parseNode node;
      entry = entryFor parsed.name;
    in
    if parsed.extra == null then
      liveDependencies entry
    else
      [ parsed.name ]
      ++ liveDependencies {
        dependencies =
          entry.optional-dependencies.${parsed.extra}
            or (throw "uv.lock: ${parsed.name} has no extra ${parsed.extra}");
      };
  liveDependencies =
    entry:
    lib.concatMap (d: if d ? extra then map (e: "${d.name}[${e}]") d.extra else [ d.name ]) (
      lib.filter (d: !(d ? marker) || markerHolds d.marker) (entry.dependencies or [ ])
    );

  locked =
    name:
    let
      entry = entryFor name;
      wheel =
        lib.findFirst compatible (throw "uv.lock has no ${name} wheel for ${cpython} on this platform")
          entry.wheels;
    in
    pythonPackages.buildPythonPackage {
      pname = name;
      inherit (entry) version;
      format = "wheel";
      src = pkgs.fetchurl { inherit (wheel) url hash; };
      # Wheels bundle native code (pydantic-core, cryptography, the Claude
      # CLI in claude-agent-sdk): point its ELF objects at nixpkgs libraries.
      nativeBuildInputs = lib.optionals stdenv.hostPlatform.isLinux [ pkgs.autoPatchelfHook ];
      buildInputs = lib.optionals stdenv.hostPlatform.isLinux [ stdenv.cc.cc.lib ];
      # The closure is flattened into jake-tools below, from uv.lock.
      dontCheckRuntimeDeps = true;
      doCheck = false;
    };

  package = pythonPackages.buildPythonPackage {
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
    # pyproject pins the uv_build series uv itself uses for development;
    # nixpkgs ships a newer one that builds the same wheel.
    pypaBuildFlags = [ "--skip-dependency-check" ];
    dependencies = map locked closure;
    dontCheckRuntimeDeps = true;
    doCheck = false;

    # Runs against the installed package with its full dependency closure.
    pythonImportsCheck = [
      "jake_tools"
      "jake_tools.mcp.server"
    ];

    passthru.tests.smoke = pkgs.runCommand "jake-tools-smoke" { nativeBuildInputs = [ package ]; } ''
      jake-tools --help > /dev/null
      jake-tools-mcp --help > /dev/null
      touch $out
    '';

    meta.mainProgram = "jake-tools-mcp";
  };
in
package
