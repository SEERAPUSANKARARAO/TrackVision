from setuptools import find_packages,setup

setup(
    name='TrackVision',
    version='0.0.1',
    author='sankar',
    author_email='seerapusankar5@gmail.com',
    install_requires=["opencv-python", "pandas", "ultralytics", "cvzone", "Pillow", "numpy", "streamlit"],
    packages=find_packages()
)