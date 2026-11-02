# The pinned FluidAudio models for transcription-service, as an immutable
# models root for `serve --models-dir`: the same tree `provision` writes,
# fetched file by file from Hugging Face at the pinned revisions.
#
# transcription-models.json lists every file `provision` downloads, with its
# SHA-256. Regenerate it after bumping `ModelPins` or FluidAudio: provision
# into an empty directory and hash the result. The `.fluidaudio-revision`
# markers are written here, as `provision` does, so offline loading accepts
# the directories without writing to them.
{
  lib,
  fetchurl,
  runCommand,
}:

let
  repos = lib.importJSON ./transcription-models.json;

  fetchFile =
    repo: file:
    fetchurl {
      name = lib.strings.sanitizeDerivationName (baseNameOf file.path);
      url = "https://huggingface.co/${repo.repo}/resolve/${repo.revision}/${file.path}";
      inherit (file) hash;
    };

  installRepo = repo: ''
    ${lib.concatMapStrings (file: ''
      install -Dm444 ${fetchFile repo file} "$out/${repo.folder}/${file.path}"
    '') repo.files}
    echo ${repo.revision} > "$out/${repo.folder}/.fluidaudio-revision"
  '';
in
runCommand "transcription-models"
  {
    passthru.revisions = lib.listToAttrs (map (repo: lib.nameValuePair repo.repo repo.revision) repos);
  }
  ''
    ${lib.concatMapStrings installRepo repos}
  ''
