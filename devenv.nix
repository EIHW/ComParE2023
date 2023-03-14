{ pkgs, eihw-packages, config, lib, self, ... }:
let
  stable-pkgs = import eihw-packages.inputs.nixpkgs {
    system = pkgs.system;
    overlays = [eihw-packages.overlays.default];
    config = {
      allowUnfree = true;
      cudaSupport = false;
    };
  };
in
rec {
  name = "ComParE23";

  packages = [
    pkgs.eihw-packages.opensmile
    (pkgs.writeShellScriptBin "deepspectrum" "LD_LIBRARY_PATH=${stable-pkgs.stdenv.cc.cc.lib}/lib:${stable-pkgs.glibc}/lib:$LD_LIBRARY_PATH ${stable-pkgs.eihw-packages.deepspectrum}/bin/deepspectrum \"$@\"")
    (pkgs.writeShellScriptBin "audeep" "LD_LIBRARY_PATH=${stable-pkgs.stdenv.cc.cc.lib}/lib:${stable-pkgs.glibc}/lib:$LD_LIBRARY_PATH ${stable-pkgs.eihw-packages.audeep}/bin/audeep \"$@\"")
    pkgs.blas.dev
    pkgs.gcc.out
    pkgs.glibc
    pkgs.lapack.dev
    pkgs.libsndfile.out
    pkgs.llvm.dev
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.jre_minimal
  ];
  languages.python = {
    enable = true;
    package = pkgs.python310;
    poetry = {
      enable = !(config.container.isBuilding);
      package =  pkgs.poetry.override {
        python3 = pkgs.python310;
      };
    };
  };
  env.DEVENV_STATE = lib.mkIf (config.container.isBuilding) (lib.mkForce ".devenv/state");
  env.DEVENV_DOTFILE = lib.mkIf (config.container.isBuilding) (lib.mkForce ".devenv");


  # https://devenv.sh/scripts/
  containers.shell = {
    registry = "docker://docker.io/mauricege/";
    copyToRoot = ./.devenv/state;
  };

  enterShell = lib.mkIf (config.container.isBuilding) ''
    rm venv/bin/python
    python -m venv --upgrade $(${pkgs.coreutils}/bin/readlink venv)
    ${pkgs.findutils}/bin/find ./venv/bin/* -type f -exec ${pkgs.gnused}/bin/sed -i "1s~^#!${builtins.getEnv "PWD"}/.devenv/state/venv/bin/python~#!/venv/bin/python~" {} \;

    VIRTUAL_ENV="/venv"
    export VIRTUAL_ENV
    PATH="$VIRTUAL_ENV/bin:$PATH"
    export PATH
  '';
}
