# Unsized enums

Rust does not support unsized (`?Sized`) variants in an enum.  This
crate provides an unsized enum with one unsized variant and one sized
variant, returned boxed along with a common base structure.  The enum
may be read and modified, including switching variants, even through a
trait object reference.

### Documentation

See the [crate documentation](http://docs.rs/unsized_enum).

# License

This project is licensed under either the Apache License version 2 or
the MIT license, at your option.  (See
[LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT)).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

