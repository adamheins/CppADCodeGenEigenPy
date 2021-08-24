def pytest_addoption(parser):
    parser.addoption("--builddir", action="store", default="build")
