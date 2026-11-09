"""
Shortcut script để chạy GVHMR batch trên toàn bộ thư mục input.
Tự động gán nhãn aria01/aria02 dựa theo GT bbox, bỏ qua render video để tối đa tốc độ.
"""
from tools.demo.demo_folder import main

if __name__ == "__main__":
    main()

