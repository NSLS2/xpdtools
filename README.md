# XPD Tools [![CI](https://github.com/NSLS2/xpd-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/NSLS2/xpd-tools/actions/workflows/ci.yml)

Tools for NSLS-II XPD beamline.

## X-ray/UV-Vis optimization

Install the optional optimization stack with:

```console
pip install "xpd-tools[optimization]"
```

PDF phase references are supplied at runtime instead of bundled with the package.
Create a version-1 JSON file whose relative paths are resolved beside the file:

```json
{
  "schema_version": 1,
  "phases": [
    {
      "name": "CsPbBr3",
      "gr_path": "CsPbBr3.gr",
      "cif_path": "CsPbBr3.cif",
      "minimize": false,
      "constraint_profile": "cs_pb_br3"
    }
  ]
}
```

Evaluate completed runs with `xpd-evaluate-xray-uvvis --pdf-references PATH UID`.
Queue Server deployments bind hardware through
`xpd_tools.optimization.plans.create_xray_uvvis_plan`; importing
`xpd_tools.optimization` itself does not connect to hardware or services.
