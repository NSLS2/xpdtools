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
The evaluator accepts raw and fitted PDF modes, and the Queue Server agent uses the
same loaded phase schema:

```python
from xpd_tools.optimization.agent import build_queue_agent
from xpd_tools.optimization.evaluation import XrayUvvisEvaluation

evaluator = XrayUvvisEvaluation(
    raw_client,
    sandbox_client,
    "references.json",
    pdf_mode="fit",  # or "raw"
)
agent = build_queue_agent(
    evaluator,
    re_manager_api,
    document_dispatcher,
    checkpoint_path="optimization-checkpoint.json",
)
```

Queue Server workers bind hardware and process settings once. The standard
CsPb/Br/I2 setup, with pre- and post-equilibrium dilution and one wash cycle,
is represented directly:

```python
from xpd_tools.optimization.plans import (
    DilutionStage,
    FlowSource,
    WashCycle,
    XrayUvvisPlanContext,
    create_xray_uvvis_plan,
)

context = XrayUvvisPlanContext(
    qepro=qepro,
    led=led,
    uv_shutter=uv_shutter,
    fast_shutter=fast_shutter,
    xray_detector=pe1c,
    wrap_xray_run=dark_plan,
    sources=(
        FlowSource(
            dof="infusion_rate_CsPb",
            pump=dds2_p1,
            precursor="CsPbOA",
            sample_label="CsPb",
        ),
        FlowSource(
            dof="infusion_rate_Br",
            pump=dds2_p2,
            precursor="TOABr",
            sample_label="Br",
        ),
        FlowSource(
            dof="infusion_rate_I2",
            pump=dds3_p1,
            precursor="ZnI2",
            sample_label="I2",
        ),
    ),
    dilutions=(
        DilutionStage(
            pump=dds1_p1,
            ratio=1.0,
            position="before_equilibrium",
            syringe_ml=20,
            material="plastic_BD",
            target_ml=20,
        ),
        DilutionStage(
            pump=ultra2,
            ratio=1.0,
            position="after_equilibrium",
            syringe_ml=100,
            material="steel",
            target_ml=100,
            wait_sec=30,
        ),
    ),
    wash_cycles=(WashCycle(pump=ultra1),),
)
xray_uvvis_acquire = create_xray_uvvis_plan(context)
```

Pass `dilutions=()` or `wash_cycles=()` explicitly when either stage is absent.
Importing `xpd_tools.optimization` itself does not connect to hardware or services.
