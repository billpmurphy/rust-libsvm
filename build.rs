fn main() {
    cc::Build::new()
        .file("libsvm/svm.cpp")
        .flag("-std=c++11")
        .compile("libsvm.a");
}
