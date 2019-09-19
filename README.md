# 起動手順

## 初回セットアップ

レポジトリをclonseする.

```
$ git clone https://github.com/miyosuda/animalai.git
```

```
$ cd animalai
$ ./scripts/setup.sh
$ ./scripts/build.sh
```
でDocker Imageを作成.


学習用にコードを変更する.

# 2回目以降

```
$ screen
```

でscreen起動


```
$ sudo docker-compose up -d
$ ./scripts/exec.sh
```

で起動.


```
$ sudo docker-compose down
```

で終了


## Tensorboard の起動

ローカルでもう一つターミナルを開いて ssh でインスタンスにログイン.

```
$ ssh ubuntu@IP
```

```
$ screen
$ cd animalai
$ ./scripts/exec.sh
```

Tensorboard 起動.

```
$ cd train
$ tensorboard --logdir=summaries
```

結果を確認する時は別コンソールから

```
$ ssh ubuntu@IP -L 6006:localhost:6006
```

