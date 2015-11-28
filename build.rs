extern crate gcc;

fn main() {
    println!("cargo:rustc-flags=-l dylib=stdc++");
    gcc::compile_library("libsvm.a", &["libsvm/svm.cpp"]);
}
