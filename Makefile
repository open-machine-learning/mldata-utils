VERSION=$(shell awk --field-separator "'" '/VERSION/{print $$2;exit}' ml2h5/__init__.py)
TARBALL="ml2h5-$(VERSION).tar.bz2"
RELEASES="../releases"

dist:
	@python setup.py sdist --force-manifest --dist-dir=$(RELEASES) --formats=bztar

all: dist
.PHONY: dist
