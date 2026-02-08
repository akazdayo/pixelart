{
  description = "pixelart streamlit app with uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        python = pkgs.python312;

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          );

        streamlitEnv = pythonSet.mkVirtualEnv "pixelart-env" {
          streamlit = [];
          opencv-python-headless = [];
          numpy = [];
          pillow = [];
          scikit-learn = [];
          pixelart-modules = [];
          pandas = [];
          pytest = [];
          "pytest-cov" = [];
        };

        mainRunner = pkgs.writeShellApplication {
          name = "main";
          runtimeInputs = [streamlitEnv];
          text = ''
            cd "${./.}"
            exec streamlit run main.py "$@"
          '';
        };

        testRunner = pkgs.writeShellApplication {
          name = "test";
          runtimeInputs = [streamlitEnv];
          text = ''
            exec pytest "$@"
          '';
        };
      in {
        packages = {
          default = mainRunner;
          env = streamlitEnv;
          test = testRunner;
        };

        apps = {
          default = {
            type = "app";
            program = "${mainRunner}/bin/main";
          };
          test = {
            type = "app";
            program = "${testRunner}/bin/test";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.alejandra
            streamlitEnv
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = pythonSet.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            unset PYTHONPATH
          '';
        };
      }
    );
}
