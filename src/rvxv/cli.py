"""RVXV command-line interface.

Usage:
    rvxv generate --spec <yaml> --output <dir> [--targets spike,tests,assertions,docs]
    rvxv validate --spec <yaml>
    rvxv preset --list
    rvxv preset --name <preset_name> --output <dir>
    rvxv schema --output <path>
"""

from __future__ import annotations

from pathlib import Path

import click

from rvxv import __version__


@click.group()
@click.version_option(version=__version__, prog_name="rvxv")
def main() -> None:
    """RVXV — RISC-V AI Extension Verification Platform.

    Verify custom RISC-V AI instructions in minutes, not months.
    """


@main.command()
@click.option("--spec", required=True, type=click.Path(exists=True), help="YAML instruction spec")
@click.option("--output", required=True, type=click.Path(), help="Output directory")
@click.option(
    "--targets",
    default="spike,tests,assertions,docs",
    help="Comma-separated targets: spike,tests,assertions,docs,rtl",
)
@click.option("--dut-top", default="core_top", help="DUT top module name for bind file")
@click.option("--clk-signal", default="clk", help="Clock signal name in DUT")
@click.option("--rst-signal", default="rst_n", help="Active-low reset signal name")
def generate(
    spec: str, output: str, targets: str,
    dut_top: str, clk_signal: str, rst_signal: str,
) -> None:
    """Generate verification artifacts from an instruction specification."""
    from rvxv.core.spec_parser import load_spec

    try:
        spec_path = Path(spec)
        output_dir = Path(output)
        target_list = [t.strip() for t in targets.split(",")]

        click.echo(f"Loading spec: {spec_path}")
        specs = load_spec(spec_path)
        click.echo(f"Parsed {len(specs)} instruction(s):")
        for s in specs:
            click.echo(f"  - {s.name}: {s.description}")

        generated_files: list[Path] = []

        if "spike" in target_list:
            click.echo("\nGenerating Spike extensions...")
            from rvxv.generators.spike.spike_gen import SpikeGenerator

            gen = SpikeGenerator()
            files = gen.generate(specs, output_dir)
            generated_files.extend(files)
            click.echo(f"  Generated {len(files)} file(s)")

        if "tests" in target_list:
            click.echo("\nGenerating test suites...")
            from rvxv.generators.tests.test_gen import AssemblyTestGenerator

            gen = AssemblyTestGenerator()
            files = gen.generate(specs, output_dir)
            generated_files.extend(files)
            click.echo(f"  Generated {len(files)} file(s)")

        if "assertions" in target_list:
            click.echo("\nGenerating assertions...")
            from rvxv.generators.assertions.assertion_gen import AssertionGenerator

            gen = AssertionGenerator(
                dut_top=dut_top,
                clk_signal=clk_signal,
                rst_signal=rst_signal,
            )
            files = gen.generate(specs, output_dir)
            generated_files.extend(files)
            click.echo(f"  Generated {len(files)} file(s)")

        if "docs" in target_list:
            click.echo("\nGenerating documentation...")
            from rvxv.generators.docs.spec_doc_gen import DocGenerator

            gen = DocGenerator()
            files = gen.generate(specs, output_dir)
            generated_files.extend(files)
            click.echo(f"  Generated {len(files)} file(s)")

        if "rtl" in target_list:
            click.echo("\nGenerating RTL functional models...")
            from rvxv.generators.rtl.rtl_gen import RTLGenerator

            gen = RTLGenerator()
            files = gen.generate(specs, output_dir)
            generated_files.extend(files)
            click.echo(f"  Generated {len(files)} file(s)")

        click.echo(f"\nTotal: {len(generated_files)} files generated in {output_dir}/")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--spec", required=True, type=click.Path(exists=True), help="YAML instruction spec")
def validate(spec: str) -> None:
    """Validate an instruction specification YAML file."""
    from rvxv.core.spec_parser import load_spec

    try:
        specs = load_spec(Path(spec))
        click.echo(f"Valid: {len(specs)} instruction(s) parsed successfully")
        for s in specs:
            click.echo(f"  - {s.name}: {s.description}")
            click.echo(f"    Encoding: {s.encoding.format}, opcode=0x{s.encoding.opcode:02x}")
            click.echo(
                f"    MATCH=0x{s.encoding.match_value:08X}"
                f" MASK=0x{s.encoding.mask_value:08X}"
            )
            click.echo(f"    Operation: {s.semantics.operation.value}")
            operands_str = ", ".join(
                f"{k}({v.element.value})" for k, v in s.operands.items()
            )
            click.echo(f"    Operands: {operands_str}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--list", "list_presets", is_flag=True, help="List available presets")
@click.option("--name", "preset_name", type=str, help="Preset name to generate")
@click.option("--output", type=click.Path(), help="Output directory")
@click.option("--targets", default="spike,tests,assertions,docs", help="Comma-separated targets")
def preset(list_presets: bool, preset_name: str | None, output: str | None, targets: str) -> None:
    """Use pre-built instruction specification presets."""
    from rvxv.presets.registry import list_presets as get_presets
    from rvxv.presets.registry import load_preset

    if list_presets:
        click.echo("Available presets:")
        for name, desc in get_presets().items():
            click.echo(f"  {name:20s} — {desc}")
        return

    if preset_name is None:
        click.echo("Error: specify --name <preset> or --list", err=True)
        raise SystemExit(1)

    if output is None:
        click.echo("Error: --output is required with --name", err=True)
        raise SystemExit(1)

    output_dir = Path(output)
    target_list = [t.strip() for t in targets.split(",")]

    try:
        click.echo(f"Loading preset: {preset_name}")
        specs = load_preset(preset_name)
        click.echo(f"Loaded {len(specs)} instruction(s)")

        generated_files: list[Path] = []

        if "spike" in target_list:
            from rvxv.generators.spike.spike_gen import SpikeGenerator

            files = SpikeGenerator().generate(specs, output_dir)
            generated_files.extend(files)

        if "tests" in target_list:
            from rvxv.generators.tests.test_gen import AssemblyTestGenerator

            files = AssemblyTestGenerator().generate(specs, output_dir)
            generated_files.extend(files)

        if "assertions" in target_list:
            from rvxv.generators.assertions.assertion_gen import AssertionGenerator

            files = AssertionGenerator().generate(specs, output_dir)
            generated_files.extend(files)

        if "docs" in target_list:
            from rvxv.generators.docs.spec_doc_gen import DocGenerator

            files = DocGenerator().generate(specs, output_dir)
            generated_files.extend(files)

        if "rtl" in target_list:
            from rvxv.generators.rtl.rtl_gen import RTLGenerator

            files = RTLGenerator().generate(specs, output_dir)
            generated_files.extend(files)

        click.echo(f"Generated {len(generated_files)} files in {output_dir}/")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--output", required=True, type=click.Path(), help="Output path for JSON Schema")
def schema(output: str) -> None:
    """Export JSON Schema for instruction specifications.

    Useful for editor autocompletion in YAML spec files.
    Add to your YAML: # yaml-language-server: $schema=./instruction_spec.schema.json
    """
    from rvxv.core.spec_parser import export_json_schema

    export_json_schema(Path(output))
    click.echo(f"JSON Schema exported to: {output}")


if __name__ == "__main__":
    main()
