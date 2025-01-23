# stablehlo-mlir

---

설정 법 : 

```sh
$ git clone --recursive https://github.com/Piorosen/stablehlo-mlir
$ cd stablehlo-mlir
$ docker build -t mlir .
$ docker run -d -p 8888:22 -v $(pwd)/app mlir
$ ssh root@localhost -p 8888
# cd /app && ./setup.sh 
```
