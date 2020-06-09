# VpPy
VpPy is a package for estimating vanishing points on images 

## Dependencies
+ numpy
+ opencv-python
+ sklearn

## Installation
```bash
python setup.py install
```

## Usage
Example: Calculate vanishing points with the kde method
```bash
vppy example/input example/output --method kde
```
![final_kde](example/output/example_final_kde.jpg)

Example: Calculate vanishing points with the intersection method
```bash
vppy example/input example/output --method intersection
```
![final_kde](example/output/example_final_intersection.jpg)

Read help for all parameters and settings available 
```bash
vppy -h
```

## Contributors

- [Erwin van Duijnhoven](https://github.com/evduijnhoven)


## License
[MIT](https://choosealicense.com/licenses/mit/)