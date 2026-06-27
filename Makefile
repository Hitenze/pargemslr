.PHONY: all clean smoke-test standard-test test

all:
	$(MAKE) -C ParGeMSLR

clean:
	$(MAKE) -C ParGeMSLR clean

smoke-test:
	ParGeMSLR/TESTS/smoke_test.sh

standard-test:
	ParGeMSLR/TESTS/standard_test.sh

test: smoke-test
