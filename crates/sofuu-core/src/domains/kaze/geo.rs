//! Spherical geometry for spatial maths — pure functions, no host, no I/O.
//!
//! A point is `[lat_degrees, lon_degrees, altitude_m]`. Distances are metres on
//! the WGS84 mean sphere (the Haversine formula), which is within about 0.5% of
//! the true geodesic for the ranges a field script cares about — and unlike a
//! flat-earth approximation it stays right over kilometres.
//!
//! Altitude is deliberately *not* folded into `distance`; a horizontal distance
//! is what a geofence or a flight-plan check wants, and a script that needs the
//! 3-D length of a local offset says `length of (target - home)`.

use std::f64::consts::PI;

/// WGS84 mean radius, metres.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Great-circle distance in metres between two lat/lon points.
pub fn haversine_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (lat1, lon1) = (a[0].to_radians(), a[1].to_radians());
    let (lat2, lon2) = (b[0].to_radians(), b[1].to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * h.sqrt().min(1.0).asin()
}

/// Great-circle distance plus the altitude difference, in metres.
///
/// This is the distance a vehicle actually flies between two points when it
/// changes height on the way: `distance between` is the ground track, this is
/// the slant range.
pub fn slant_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let ground = haversine_m(a, b);
    let dz = b[2] - a[2];
    (ground * ground + dz * dz).sqrt()
}

/// Initial bearing in degrees (0 = north, 90 = east) from `a` to `b`.
pub fn bearing_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (lat1, lat2) = (a[0].to_radians(), b[0].to_radians());
    let dlon = (b[1] - a[1]).to_radians();
    let y = dlon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    normalise_deg(y.atan2(x).to_degrees())
}

/// The compass bearing of a local vector, degrees clockwise from north.
///
/// A Kaze vector is `[x, y, z]` in a local frame: x east, y north, z up. This is
/// what `bearing to v` means — the direction the vector points, not the direction
/// to a place.
pub fn bearing_of(v: [f64; 3]) -> f64 {
    normalise_deg(v[0].atan2(v[1]).to_degrees())
}

/// Euclidean length of a vector, in whatever unit its components are in.
pub fn magnitude(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalise_deg(deg: f64) -> f64 {
    let d = deg % 360.0;
    if d < 0.0 {
        d + 360.0
    } else {
        d
    }
}

/// Degrees to radians. `x in radians` in the language.
pub fn to_radians(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Radians to degrees. `x in degrees` in the language.
pub fn to_degrees(rad: f64) -> f64 {
    rad * 180.0 / PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn one_degree_of_latitude_is_about_111_km() {
        let d = haversine_m([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(close(d, 111_194.9, 50.0), "{d}");
    }

    #[test]
    fn one_degree_of_longitude_shrinks_with_latitude() {
        let equator = haversine_m([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let zurich = haversine_m([47.6, 8.5, 0.0], [47.6, 9.5, 0.0]);
        assert!(close(equator, 111_194.9, 50.0), "{equator}");
        assert!(zurich < equator, "{zurich} should be shorter than {equator}");
        assert!(close(zurich, 75_000.0, 2_000.0), "{zurich}");
    }

    #[test]
    fn a_short_hop_across_a_swiss_field_is_tens_of_metres() {
        let a = [47.3769, 8.5417, 408.0];
        let b = [47.3770, 8.5420, 410.0];
        let d = haversine_m(a, b);
        assert!(d > 20.0 && d < 30.0, "{d} m across the field");
    }

    #[test]
    fn bearing_has_the_cardinal_directions_right() {
        let here = [0.0, 0.0, 0.0];
        let north = [1.0, 0.0, 0.0];
        let east = [0.0, 1.0, 0.0];
        let south = [-1.0, 0.0, 0.0];
        let west = [0.0, -1.0, 0.0];
        assert!(close(bearing_deg(here, north), 0.0, 0.01));
        assert!(close(bearing_deg(here, east), 90.0, 0.01));
        assert!(close(bearing_deg(here, south), 180.0, 0.01));
        assert!(close(bearing_deg(here, west), 270.0, 0.01));
    }

    #[test]
    fn a_northbound_bearing_never_reports_360() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let brg = bearing_deg(a, b);
        assert!(brg >= 0.0 && brg < 360.0, "{brg}");
    }

    #[test]
    fn a_local_vector_has_a_compass_bearing() {
        assert!(close(bearing_of([0.0, 1.0, 0.0]), 0.0, 0.01));
        assert!(close(bearing_of([1.0, 0.0, 0.0]), 90.0, 0.01));
        assert!(close(bearing_of([0.0, -1.0, 0.0]), 180.0, 0.01));
        assert!(close(bearing_of([-1.0, 0.0, 0.0]), 270.0, 0.01));
        assert!(close(bearing_of([1.0, 1.0, 0.0]), 45.0, 0.01));
    }

    #[test]
    fn magnitude_is_the_3d_length() {
        assert!(close(magnitude([3.0, 4.0, 12.0]), 13.0, 0.001));
        assert!(close(magnitude([0.0, 0.0, 0.0]), 0.0, 0.0001));
    }

    #[test]
    fn a_climb_makes_the_slant_range_longer_than_the_ground_track() {
        let a = [47.0, 8.0, 100.0];
        let b = [47.0, 8.01, 200.0];
        let ground = haversine_m(a, b);
        let slant = slant_m(a, b);
        assert!(slant > ground, "slant {slant} > ground {ground}");
        let expected = (ground * ground + 10000.0).sqrt();
        assert!(close(slant, expected, 0.01));
    }

    #[test]
    fn the_same_point_is_zero_apart() {
        let p = [47.3769, 8.5417, 408.0];
        assert!(close(haversine_m(p, p), 0.0, 0.0001));
        assert!(close(slant_m(p, p), 0.0, 0.0001));
    }

    #[test]
    fn a_degree_is_pi_over_180_radians() {
        assert!(close(to_radians(180.0), PI, 0.000001));
        assert!(close(to_degrees(PI), 180.0, 0.000001));
    }
}
