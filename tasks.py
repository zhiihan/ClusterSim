from invoke import task


@task
def clean(c):
    patterns = ["build", "**/*.pyc", "dist", "docs/build/"]
    for pattern in patterns:
        c.run(f"rm -rf {pattern}")


@task(clean)
def docs(c, serve=False):
    c.run("sphinx-apidoc -o docs/api/ --module-first --force src/cluster_sim")

    # MacOS : compile with
    # clang++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) bindings.cpp graphsim.cpp loccliff.cpp stabilizer.cpp -o graphsim$(python3-config --extension-suffix)

    if serve:
        c.run("sphinx-autobuild --open-browser docs/ docs/build/")
    else:
        c.run("sphinx-build docs/ docs/build/")
