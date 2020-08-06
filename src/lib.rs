//! # Rust unsized enum implementation
//!
//! In stable Rust as of mid-2020, `?Sized` (aka unsized or trait
//! object or "dynamically sized type" or DST) support is missing in
//! various places:
//!
//! - Rust's built-in enums don't support `?Sized` variants
//!
//! - `Option` doesn't support `?Sized` (because it is an enum)
//!
//! - Rust's `union` type doesn't support `?Sized`
//!
//! - `MaybeUninit` also doesn't support `?Sized` (because it is
//! currently built on top of `union`)
//!
//! So if there is a requirement for unsized types within an enum,
//! some other approach must be taken.  Currently this crate
//! implements a single enum ([`UnsizedEnum`]), with one unsized
//! variant and one sized variant, which is always returned boxed.
//! (This approach can only support a *single* unsized variant,
//! although it could be extended to provide additional *sized*
//! variants.)
//!
//! The enum may be read and modified, including switching variants,
//! even through a trait object reference.
//!
//! # Safety and soundness discussion
//!
//! This crate is intended to be sound, and if unsoundness can be
//! demonstrated, it will be fixed (if possible) or else the API
//! marked as unsafe until a safe way can be found.  However right now
//! the theoretical (rather than practical) soundness of the crate
//! depends on aspects of Rust's safety contract which are as yet
//! undecided.  So if that makes you nervous, then don't use this
//! crate for the time being.
//!
//! The boxed memory is accessed with two structures which represent
//! two views of that memory: `UnsizedEnum` for the header plus the
//! `V0` (unsized) variant, and `UnsizedEnum_V1` for the header plus
//! the `V1` (sized) variant.  `#[repr(C)]` is used to enforce the
//! order of members and to ensure that the header part of the
//! `UnsizedEnum` structure is compatible with `UnsizedEnum_V1`.
//!
//! It's necessary to include the `V0` instance directly in the
//! `UnsizedEnum` structure, because its cleanup must be handled
//! through the vtable.  If no `V0` value is included in
//! `UnsizedEnum`, it seems that the Drop handler doesn't receive a
//! fat pointer, and so has no access to the vtable.  However in the
//! case of storing a `V1` variant, the `V0` value included in the
//! `UnsizedEnum` must not be dropped because it will be invalid data
//! for the `V0` type.  So the `V0` value is made `ManuallyDrop` so
//! that we can skip dropping that invalid `V0` data in the `V1`
//! variant case.  (`MaybeUninit` would be better but it doesn't
//! support `?Sized` yet.)
//!
//! So strictly speaking in the case of storing the `V1` variant,
//! because the `UnsizedEnum` struct contains `val: ManuallyDrop<V0>`,
//! we're working with references to an invalid `UnsizedEnum` (invalid
//! in the `val` part).  However we never "produce" an invalid
//! `UnsizedEnum` value.  `V1` values are produced using
//! `UnsizedEnum_V1`.  The only code that is exposed to the entire
//! invalid `UnsizedEnum` is the compiler-generated drop code.
//! (Whether passing around a reference to invalid data is
//! theoretically sound or not is undecided, but it seems like the
//! [consensus is leaning towards it being
//! sound](https://github.com/rust-lang/unsafe-code-guidelines/issues/77).)
//!
//! It's important that the niche-filling optimisation doesn't try to
//! make use of any unused bit-patterns in the `V0` value to store
//! data, because those may overwrite the value for the `V1` variant.
//! However since this implementation is in total control of the
//! structure and the structure is returned boxed, there is no way for
//! a crate user to cause the `ManuallyDrop<V0>` value to be wrapped
//! in an `enum`, so there should be no case where niche-filling would
//! try to make use of the memory within the `ManuallyDrop`.  So the
//! compiler-generated drop code should have no reason to touch the
//! `V0` variant memory in the `V1` case.  So our Drop implementation
//! is free to skip dropping the (invalid) `V0` value and drop the
//! overlaid `V1` value instead.
//!
//! If Rust gains support for `?Sized` in more places, especially
//! `MaybeUninit`, this implementation will be improved to use those
//! features.  That would also resolve the question about the
//! theoretical soundness of holding a reference to invalid data.
//!
//! In addition it's necessary to compare vtable pointers in `set_v0`.
//! This depends on the layout of fat pointers.  This is much
//! lower-risk, since it will fail immediately and very obviously in
//! testing if the layout changes in the compiler.  Also several other
//! crates already depend on this layout.
//!
//! [`UnsizedEnum`]: struct.UnsizedEnum.html

use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem;
use std::mem::ManuallyDrop;
use std::ptr;

/// An unsized enum with two variants
///
/// As this is a DST, this enum can only be created on the heap.  So
/// for convenience, it allows a common base structure to be included
/// in the header before the enum variants.  Also, since a lot of
/// space will generally be wasted by the discriminant due to
/// alignment and packing (even if it is only a `u8`), a whole `usize`
/// is used and the higher bits in the discriminant are made available
/// to the caller for storage.
///
/// The type parameters are: the common base type, the first variant
/// (which may be unsized) and the second variant (which must be
/// sized).
#[repr(C)]
pub struct UnsizedEnum<B, V0: ?Sized, V1> {
    header: Header<B>,
    phantomdata: PhantomData<V1>,
    val: ManuallyDrop<V0>,
}

#[repr(C)]
struct UnsizedEnum_V1<B, V0: ?Sized, V1> {
    header: Header<B>,
    val: V1,
    phantomdata: PhantomData<V0>,
}

#[repr(C)]
struct Header<B> {
    disc: usize,
    base: B,
}

// const max hack
#[allow(dead_code)]
const fn max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

// If a non-fat pointer is passed, returns a null pointer
#[inline(always)]
fn vtable_of<T: ?Sized>(fat: &T) -> *const () {
    #[repr(C)]
    struct FatPointer {
        data: *const (),
        meta: *const (),
    }
    if mem::size_of_val(&fat) < mem::size_of::<FatPointer>() {
        std::ptr::null()
    } else {
        assert_eq!(mem::size_of_val(&fat), mem::size_of::<FatPointer>());
        let repr = unsafe { mem::transmute_copy::<&T, FatPointer>(&fat) };
        repr.meta
    }
}

impl<B, V0, V1> UnsizedEnum<B, V0, V1> {
    // Safe because we're using the largest alignment and the largest
    // size.  These will be valid values for the Layout because they
    // come from existing types.  So either we're using the size and
    // alignment from one type, or the size from one type and a larger
    // alignment from the other.  Overaligning a type is safe.
    #[allow(dead_code)]
    const LAYOUT: Layout = unsafe {
        Layout::from_size_align_unchecked(
            max(
                mem::size_of::<UnsizedEnum<B, V0, V1>>(),
                mem::size_of::<UnsizedEnum_V1<B, V0, V1>>(),
            ),
            max(
                mem::align_of::<UnsizedEnum<B, V0, V1>>(),
                mem::align_of::<UnsizedEnum_V1<B, V0, V1>>(),
            ),
        )
    };

    // Returned memory is uninitialised
    unsafe fn alloc() -> *mut UnsizedEnum<B, V0, V1> {
        let p = std::alloc::alloc(Self::LAYOUT) as *mut UnsizedEnum<B, V0, V1>;
        if p.is_null() {
            std::alloc::handle_alloc_error(Self::LAYOUT);
        }
        p
    }

    /// Create a new instance of the V0 type
    pub fn new_v0(base: B, val: V0) -> Box<Self> {
        let inner = UnsizedEnum {
            header: Header { disc: 0, base },
            phantomdata: PhantomData,
            val: ManuallyDrop::new(val),
        };
        // Safe because all we're doing is writing a valid value into
        // a (possibly) bigger piece of memory.
        unsafe {
            let p = Self::alloc();
            ptr::write(p, inner);
            Box::from_raw(p)
        }
    }

    /// Create a new instance of the V1 type
    pub fn new_v1(base: B, val: V1) -> Box<Self> {
        let inner = UnsizedEnum_V1 {
            header: Header { disc: 1, base },
            val,
            phantomdata: PhantomData,
        };
        // Safe because the rest of the code uses `disc` to decide how
        // to interpret the `val` part of the structure
        unsafe {
            let p = Self::alloc();
            ptr::write(p as *mut UnsizedEnum_V1<B, V0, V1>, inner);
            Box::from_raw(p)
        }
    }
}

impl<B, V0: ?Sized, V1> UnsizedEnum<B, V0, V1> {
    /// Return a mutable reference to the common base structure `B`
    pub fn base(&mut self) -> &mut B {
        &mut self.header.base
    }

    /// Set a value in the spare bits above the discriminant.  Bit 0
    /// of this value will not be stored.
    pub fn set_spare(&mut self, spare: usize) {
        self.header.disc = (self.header.disc & 1) | (spare & !1);
    }

    /// Gets the discriminant value including any value stored in the
    /// spare bits above it.  So bit 0 will be the disriminant (0 for
    /// `V0`, 1 for `V1`), and the remaining bits will be whatever
    /// value was saved by the most recent call to `set_spare`, or 0
    /// initially.
    pub fn get_spare(&self) -> usize {
        self.header.disc
    }

    /// Change the value stored in the enum to the provided `V0`
    /// value, dropping the previous value (of whichever variant).
    #[inline]
    pub fn set_v0(&mut self, v0: &Forget<V0>) {
        // Safe because the value is only temporarily in an
        // invalid/dropped state.  The Forget type lets us do a "move"
        // operation by copying data out of it, without it being
        // dropped twice, since the Forget type will never drop it.
        // Since the method API doesn't prevent the caller providing a
        // different underlying type, it's necessary to check the
        // vtable first.
        let p = v0.get_ref();
        assert_eq!(
            vtable_of(p),
            vtable_of(self),
            "Passed a value to set_v0() from a different underlying type"
        );
        unsafe {
            self.drop_value();
            std::ptr::copy_nonoverlapping(
                p as *const V0 as *const u8,
                (&mut *self.val) as *mut V0 as *mut u8,
                std::mem::size_of_val(&*p),
            );
            self.header.disc &= !1;
        }
    }

    /// Change the value stored in the enum to the provided V1 value,
    /// dropping the previous value (of whichever variant)
    #[inline]
    pub fn set_v1(&mut self, v1: V1) {
        // Safe because the value is only temporarily in an
        // invalid/dropped state
        unsafe {
            self.drop_value();
            let p = self as *mut UnsizedEnum<B, V0, V1>;
            let p = p as *mut UnsizedEnum_V1<B, V0, V1>;
            ptr::write(&mut (*p).val, v1);
            self.header.disc |= 1;
        }
    }

    // Leaves the value part of structure in an uninitialised state.
    // Safe because we drop according to the `disc` value
    unsafe fn drop_value(&mut self) {
        if 0 == (self.header.disc & 1) {
            ManuallyDrop::drop(&mut self.val);
        } else {
            let p = self as *mut UnsizedEnum<B, V0, V1>;
            let p = p as *mut UnsizedEnum_V1<B, V0, V1>;
            ptr::drop_in_place(&mut (*p).val);
        }
    }

    /// Get a reference to the value.  Since there are two possible
    /// variants, an enum is returned that provides either one
    /// reference or the other.
    pub fn get_mut(&mut self) -> EnumRef<V0, V1> {
        // Safe because we access the value according to `disc`.
        // Normal borrowing rules mean that no modifications can be
        // made whilst the borrow is active.
        unsafe {
            if 0 == (self.header.disc & 1) {
                EnumRef::V0(&mut *self.val)
            } else {
                let p = self as *mut UnsizedEnum<B, V0, V1>;
                let p = p as *mut UnsizedEnum_V1<B, V0, V1>;
                EnumRef::V1(&mut (*p).val)
            }
        }
    }
}

impl<B, V0: ?Sized, V1> Drop for UnsizedEnum<B, V0, V1> {
    fn drop(&mut self) {
        // Safe because we're in the drop handler, so leaving it
        // dropped is the idea.  Nothing else will drop the value part
        // because it is `ManuallyDrop`
        unsafe { self.drop_value() };
    }
}

/// A mutable reference to the active variant within the `UnsizedEnum`
pub enum EnumRef<'a, V0: ?Sized, V1> {
    V0(&'a mut V0),
    V1(&'a mut V1),
}

/// A type that guarantees not to drop the enclosed value
///
/// This is used to allow a 'move' to be executed by copying data
/// directly in unsafe code.
pub struct Forget<T: ?Sized>(ManuallyDrop<T>);

impl<T> Forget<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        Self(ManuallyDrop::new(val))
    }
}

impl<T: ?Sized> Forget<T> {
    /// Get a reference to the contained value
    pub fn get_ref(&self) -> &T {
        &(*self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{EnumRef, Forget, UnsizedEnum};
    struct Base(usize);
    struct A(u16);

    trait Sum {
        fn sum(&self) -> f64;
    }
    struct B {
        a: f64,
        b: f64,
    }
    impl Sum for B {
        fn sum(&self) -> f64 {
            self.a + self.b
        }
    }
    struct C {
        a: u32,
        b: u32,
        c: u32,
    }
    impl Sum for C {
        fn sum(&self) -> f64 {
            (self.a + self.b + self.c) as f64
        }
    }

    fn calc_sum(r: &mut Box<UnsizedEnum<Base, dyn Sum, A>>) -> f64 {
        match r.get_mut() {
            EnumRef::V0(v) => v.sum(),
            EnumRef::V1(a) => a.0 as f64,
        }
    }

    #[test]
    fn test() {
        let mut e: Box<UnsizedEnum<Base, dyn Sum, A>>;
        // Assignments to `e` below are coercions
        e = UnsizedEnum::new_v0(Base(654321), B { a: 1.0, b: 2.0 });
        assert_eq!(calc_sum(&mut e), 3.0);
        e.set_v1(A(54321));
        assert_eq!(calc_sum(&mut e), 54321.0);
        e.set_v0(&Forget::new(B { a: 3.0, b: 4.0 }));
        assert_eq!(calc_sum(&mut e), 7.0);
        e = UnsizedEnum::<Base, C, A>::new_v1(Base(654321), A(12345));
        assert_eq!(calc_sum(&mut e), 12345.0);
        e.set_v0(&Forget::new(C { a: 3, b: 4, c: 5 }));
        assert_eq!(calc_sum(&mut e), 12.0);
        e.set_v1(A(13542));
        assert_eq!(calc_sum(&mut e), 13542.0);
    }

    #[test]
    #[should_panic]
    fn test_writing_wrong_type_1() {
        let mut e: Box<UnsizedEnum<Base, dyn Sum, A>>;
        e = UnsizedEnum::<Base, C, A>::new_v1(Base(654321), A(12345));
        // Can't write a `B` to a `C` instance
        e.set_v0(&Forget::new(B { a: 3.0, b: 4.0 }));
    }

    #[test]
    #[should_panic]
    fn test_writing_wrong_type_2() {
        let mut e: Box<UnsizedEnum<Base, dyn Sum, A>>;
        e = UnsizedEnum::new_v0(Base(654321), C { a: 3, b: 4, c: 5 });
        // Can't write a `B` to a `C` instance
        e.set_v0(&Forget::new(B { a: 3.0, b: 4.0 }));
    }

    #[test]
    fn test_sized() {
        // This tests that `set_v0` also works okay when there are no
        // fat pointers involved
        let mut e = UnsizedEnum::new_v0(Base(654321), B { a: 1.0, b: 2.0 });
        e.set_v1(A(54321));
        e.set_v0(&Forget::new(B { a: 3.0, b: 4.0 }));
    }
}
