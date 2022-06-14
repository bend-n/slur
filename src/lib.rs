//! A fast, iterative, correct approach to Stackblur, resulting in a very smooth
//! and high-quality output, with no edge bleeding.
//!
//! This crate implements a tweaked version of the Stackblur algorithm requiring
//! `radius * 2 + 2` elements of space rather than `radius * 2 + 1`, which is a
//! small tradeoff for much-increased visual quality.
//!
//! The algorithm is exposed as an iterator ([`StackBlur`]) that can wrap any
//! other iterator that yields elements of [`StackBlurrable`]. The [`StackBlur`]
//! will then yield elements blurred by the specified radius.
//!
//! ## Benefits of this crate
//!
//! Stackblur is essentially constant-time. Regardless of the radius, it always
//! performs only 1 scan over the input iterator and outputs exactly the same
//! amount of elements.
//!
//! Additionally, it produces results that are comparable to slow and expensive
//! Gaussian blurs. As opposed to box blur which uses a basic rolling average,
//! Stackblur uses a weighted average where each output pixel is affected more
//! strongly by the inputs that were closest to it.
//!
//! Despite that, Stackblur does not perform much worse compared to naive box
//! blurs, and is quite cheap compared to full Gaussian blurs, at least for the
//! CPU. The implementation in this crate will most likely beat most unoptimized
//! blurs you can find on crates.io, as well as some optimized ones, and it is
//! extremely flexible and generic.
//!
//! For a full explanation of the improvements made to the Stackblur algorithm,
//! see the [`iter`] module.
//!
//! ## Comparison to the `stackblur` crate
//!
//! `stackblur` suffers from edge bleeding and flexibility problems. For
//! example, it can only operate on buffers of 32-bit integers, and expects them
//! to be packed linear ARGB pixels. Additionally, it cannot operate on a 2D
//! subslice of a buffer (like `imgref` allows for this crate), and it does not
//! offer any streaming iterators or documentation. And it also only supports
//! a blur radius of up to 255.
//!
//! ## Usage
//!
//! Aside from [`StackBlurrable`] and [`StackBlur`] which host their own
//! documentation, there are helper functions like [`blur`] and [`blur_argb`]
//! that can be used to interact with 2D image buffers, due to the fact that
//! doing so manually involves unsafe code (if you want no-copy).

#![feature(portable_simd, stmt_expr_attributes)]
#![cfg_attr(test, feature(test))]

use std::collections::VecDeque;
use std::simd::{LaneCount, SupportedLaneCount};

pub extern crate imgref;

use imgref::ImgRefMut;

#[cfg(test)]
mod test;

pub mod color;
pub mod iter;
pub mod traits;

use color::Argb;
use iter::StackBlur;
use traits::StackBlurrable;

/// Blurs a buffer, assuming one element per pixel.
///
/// The provided closures are used to convert from the buffer's native pixel
/// format to [`StackBlurrable`] values that can be consumed by [`StackBlur`].
pub fn blur<T, B: StackBlurrable>(
    buffer: &mut ImgRefMut<T>,
    radius: usize,
    mut to_blurrable: impl FnMut(&T) -> B,
    mut to_pixel: impl FnMut(B) -> T,
) {
    use imgref_iter::iter::{IterWindows, IterWindowsPtrMut};
    use imgref_iter::traits::{ImgIter, ImgIterMut, ImgIterPtrMut};

    let mut ops = VecDeque::new();

    // This is needed to avoid Undefined Behavior. Writing to the rows of the
    // must be done before constructing the columns iterators, because otherwise
    // the writes would invalidate their borrows. However I don't want to
    // duplicate this loop, so make it a closure.
    let mut blur_windows = |writer: IterWindowsPtrMut<T>, reader: IterWindows<T>| {
        for (write, read) in writer.zip(reader) {
            let mut blur = StackBlur::new(read.map(&mut to_blurrable), radius, &mut ops);
            write.for_each(|place| unsafe { *place = to_pixel(blur.next().unwrap()) });
        }
    };

    let buffer_ptr = buffer.as_mut_ptr();
    blur_windows(
        unsafe { buffer_ptr.iter_rows_ptr_mut() },
        buffer.iter_rows(),
    );
    blur_windows(
        unsafe { buffer_ptr.iter_cols_ptr_mut() },
        buffer.iter_cols(),
    );
}

/// Blurs a buffer with SIMD, assuming one element per pixel.
///
/// The provided closures are used to convert from the buffer's native pixel
/// format to [`StackBlurrable`] values that can be consumed by [`StackBlur`].
pub fn simd_blur<T, Bsimd: StackBlurrable, Bsingle: StackBlurrable, const LANES: usize>(
    buffer: &mut ImgRefMut<T>,
    radius: usize,
    mut to_blurrable_simd: impl FnMut([&T; LANES]) -> Bsimd,
    mut to_pixel_simd: impl FnMut(Bsimd) -> [T; LANES],
    mut to_blurrable_single: impl FnMut(&T) -> Bsingle,
    mut to_pixel_single: impl FnMut(Bsingle) -> T,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[cfg(not(doc))]
    use imgref_iter::iter::{
        SimdIterWindow, SimdIterWindowPtrMut, SimdIterWindows, SimdIterWindowsPtrMut,
    };
    #[cfg(not(doc))]
    use imgref_iter::traits::{ImgIterMut, ImgSimdIter, ImgSimdIterPtrMut};

    let mut ops_simd = VecDeque::new();
    let mut ops_single = VecDeque::new();

    let mut simd_blur_windows =
        |writer: SimdIterWindowsPtrMut<T, LANES>,
         reader: SimdIterWindows<T, LANES>,
         mut ops_simd: VecDeque<Bsimd>,
         mut ops_single: VecDeque<Bsingle>| {
            for (write, read) in writer.zip(reader) {
                match (write, read) {
                    (SimdIterWindowPtrMut::Simd(write), SimdIterWindow::Simd(read)) => {
                        let mut blur =
                            StackBlur::new(read.map(&mut to_blurrable_simd), radius, &mut ops_simd);
                        write.for_each(|place| {
                            place
                                .into_iter()
                                .zip(to_pixel_simd(blur.next().unwrap()))
                                .for_each(|(place, pixel)| unsafe { *place = pixel });
                        });
                    }

                    (SimdIterWindowPtrMut::Single(write), SimdIterWindow::Single(read)) => {
                        let mut blur = StackBlur::new(
                            read.map(&mut to_blurrable_single),
                            radius,
                            &mut ops_single,
                        );
                        write.for_each(|place| unsafe {
                            *place = to_pixel_single(blur.next().unwrap());
                        });
                    }

                    _ => unreachable!(),
                }
            }

            (ops_simd, ops_single)
        };

    let buffer_ptr = buffer.as_mut_ptr();
    (ops_simd, ops_single) = simd_blur_windows(
        unsafe { buffer_ptr.simd_iter_rows_ptr_mut::<LANES>() },
        buffer.simd_iter_rows::<LANES>(),
        ops_simd,
        ops_single,
    );
    simd_blur_windows(
        unsafe { buffer_ptr.simd_iter_cols_ptr_mut::<LANES>() },
        buffer.simd_iter_cols::<LANES>(),
        ops_simd,
        ops_single,
    );
}

/// Blurs a buffer of 32-bit packed ARGB pixels (0xAARRGGBB).
///
/// This is a version of [`blur`] with pre-filled conversion routines that
/// provide good results for blur radii <= 4096. Larger radii may overflow.
pub fn blur_argb(buffer: &mut ImgRefMut<u32>, radius: usize) {
    blur(buffer, radius, |i| Argb::from(*i), Argb::into);
}

/// Blurs a buffer of 32-bit packed ARGB pixels (0xAARRGGBB) with SIMD.
///
/// This is a version of [`simd_blur`] with pre-filled conversion routines that
/// provide good results for blur radii <= 4096. Larger radii may overflow.
pub fn simd_blur_argb<const LANES: usize>(buffer: &mut ImgRefMut<u32>, radius: usize)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    simd_blur(
        buffer,
        radius,
        |i: [&u32; LANES]| Argb::from(i.map(u32::clone)),
        Argb::into,
        |i| Argb::from(*i),
        Argb::into,
    );
}
