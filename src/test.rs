extern crate test;

use imgref::ImgVec;
use test::Bencher;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

#[bench]
#[inline(never)]
fn blur_argb_16(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::blur_argb(&mut buf.as_mut(), 16));
}

#[bench]
#[inline(never)]
fn blur_argb_128(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::blur_argb(&mut buf.as_mut(), 128));
}

#[bench]
#[inline(never)]
fn blur_argb_1024(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::blur_argb(&mut buf.as_mut(), 1024));
}

#[bench]
#[inline(never)]
fn simd_blur_argb_16(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::simd_blur_argb::<8>(&mut buf.as_mut(), 16));
}

#[bench]
#[inline(never)]
fn simd_blur_argb_128(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::simd_blur_argb::<8>(&mut buf.as_mut(), 128));
}

#[bench]
#[inline(never)]
fn simd_blur_argb_1024(bencher: &mut Bencher) {
    let mut buf = ImgVec::new(vec![0; WIDTH * HEIGHT], WIDTH, HEIGHT);
    bencher.iter(|| crate::simd_blur_argb::<8>(&mut buf.as_mut(), 1024));
}
