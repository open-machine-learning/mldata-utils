.PHONY: dist release all

#VERSION=$(shell awk --field-separator "'" '/VERSION/{print $$2;exit}' setup.py)
VERSION=$(shell awk -F \' '/VERSION/{print $2;exit}' setup.py)
TARBALL="mldata-utils-$(VERSION).tar.bz2"
RELEASES="../releases"
HOST=mldata.org
DIRNAME=mldata-utils-$(VERSION)

release: dist
	scp $(RELEASES)/$(TARBALL) mldata@$(HOST):tmp
	ssh mldata@$(HOST) \( cd tmp \; \
		tar xjf $(TARBALL) \;  \
		cd $(DIRNAME) \; \
		python setup.py install --prefix=/home/mldata/python/ \; \
		sudo /etc/init.d/fapws3 restart \)

dist:
	@python setup.py sdist --force-manifest --dist-dir=$(RELEASES) --formats=bztar

clean:
	find ./ -name '*.pyc' -delete
	find ./ -name '*.swp' -delete

all: dist clean
