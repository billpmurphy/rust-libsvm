extern crate gcc;

fn main() {
    gcc::compile_library("libsvm.a", &["libsvm/svm.cpp"]);
    println!("cargo:rustc-flags=-L dylib=stdc++");
}
