{
  description = "pixelart development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    utils,
    ...
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
          ];

          buildInputs = [
            pkgs.libxcb
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.libxcb
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
              pkgs.libGL
              pkgs.glib
            ]}:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}
