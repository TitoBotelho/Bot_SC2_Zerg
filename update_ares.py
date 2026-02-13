from subprocess import run

if __name__ == "__main__":
    run(["git", "submodule", "deinit", "-f", "."])
    run(["git", "submodule", "update", "--init"])
    run(["git", "submodule", "update", "--init", "--recursive", "--remote"])

    with open("ares-sc2/pyproject.toml", encoding="utf-8") as f:
        ares_contents = f.read()

    ares_contents = ares_contents.replace(
        'burnysc2 = { git = "https://github.com/august-k/python-sc2", branch = "develop" }',
        'burnysc2 = "^7.1.3"',
    )
    ares_contents = ares_contents.replace(
        'cython-extensions-sc2 = "^0.13.1"',
        'cython-extensions-sc2 = "0.13.1"',
    )

    with open("ares-sc2/pyproject.toml", "w", encoding="utf-8") as f:
        f.write(ares_contents)

    run(["poetry", "remove", "ares-sc2"])

    with open("pyproject.toml") as f:
        contents = f.readlines()

    insert_at_index = 0
    for i, l in enumerate(contents):
        if l.strip() == "[tool.poetry.dependencies]":
            insert_at_index = i + 1
            break

    dependency_line = 'ares-sc2 = { path = "ares-sc2", develop = false }\n'
    if dependency_line not in contents:
        contents.insert(insert_at_index, dependency_line)

    with open("pyproject.toml", "w") as f:
        contents = "".join(contents)
        f.write(contents)

    run(["poetry", "lock", "--no-update"])
    run(["poetry", "install"])
