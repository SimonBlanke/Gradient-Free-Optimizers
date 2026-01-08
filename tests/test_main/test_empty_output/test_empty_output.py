import os
import subprocess

here = os.path.dirname(os.path.abspath(__file__))

verbose_file = os.path.join(here, "verbose.py")
non_verbose_file = os.path.join(here, "non_verbose.py")


def test_empty_output():
    output_verbose = subprocess.run(["python", verbose_file], stdout=subprocess.PIPE)
    output_non_verbose = subprocess.run(
        ["python", non_verbose_file], stdout=subprocess.PIPE
    )

    verbose_str = output_verbose.stdout.decode()
    non_verbose_str = output_non_verbose.stdout.decode()

    print("\n verbose_str \n", verbose_str, "\n")
    print("\n non_verbose_str \n", non_verbose_str, "\n")

    assert "Results:" in verbose_str
    assert not non_verbose_str
