from src.build_killi.run01_get_hemisphere_shifts import get_hemisphere_shifts

if __name__ == '__main__':
    root = "I:\\Nick\\killi_tracker\\"
    side1_name = "20250729_LCP1-NLSMSC_side1"
    side2_name = "20250729_LCP1-NLSMSC_side2"
    interval = 10

    get_hemisphere_shifts(root=root, side1_name=side1_name, side2_name=side2_name, interval=interval, n_workers=8)

