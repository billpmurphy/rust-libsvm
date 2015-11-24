all:
	make -C libsvm lib
	cargo build

clean:
	cargo clean
	make -C libsvm clean
