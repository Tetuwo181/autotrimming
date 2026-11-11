# autotrimming

鉄道写真の **回転補正** と **オートトリミング** を学習する Keras ノートブックです。学習済みモデルは TFLite に変換してアプリ側で使う想定です。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `autotrim_training.ipynb` | Google Colab 向け学習ノートブック |
| `sample/` | 教師データ（画像 + 同名 JSON）。**Git 管理外**（`.gitignore`） |

## モデル

推論は次の2段階です。

1. **`trim_analyzer`（解析モデル）**  
   元画像 → 回転角（度）／縦横比（5クラス）／縦構図かどうか
2. **`trim_cropper`（トリミングモデル）**  
   回転後キャンバス + 指定比率 → BBox `[left, top, width, height]`（0〜1）

比率はトリミングモデル内部で厳密に守るため、出力は必ず指定比率になります。比率をユーザーが指定しない場合は、解析モデルの推定結果を使います。

対応する縦横比: `1:1` / `5:4` / `4:3` / `3:2` / `16:9`（`portrait` で縦横入れ替え可）

### バックボーン

設定セルの名前を変えるだけで切り替えられます（解析／トリミングで別指定可）。

```python
A_BACKBONE = 'MobileNetV3Large'   # 解析モデル
B_BACKBONE = 'MobileNetV3Large'   # トリミングモデル
```

候補: `MobileNetV3Small` / `MobileNetV3Large` / `EfficientNetB0`〜`B3` / `EfficientNetV2B0`〜`B3` / `EfficientNetV2S` / `MobileNetV2` / `ResNet50` / `ConvNeXtTiny`

入力は常に RGB 0〜255。正規化はモデル内で行うため、バックボーンを変えてもアプリ側の前処理は変わりません（変わるのは `A_SIZE` / `B_SIZE` を変えた場合の入力サイズだけ）。

## 教師データ

`sample/` に、画像とその同名の `.jpg.json` を 1:1 で置きます。JSON の `edit` が教師ラベルです。

| 項目 | 意味 |
|---|---|
| `rotation_degrees` | 水平にするための回転角（**正 = 時計回り**、目安 -45〜45） |
| `aspect_width` / `aspect_height` | トリミング比率（`null` は元画像比率扱い） |
| `portrait` | 縦構図なら `true`（幅と高さを入れ替え） |
| `crop` | 回転後キャンバスに対する相対比率 `{left, top, width, height}` |

回転後キャンバスのサイズは  
`W·cosθ + H·sinθ` × `W·sinθ + H·cosθ`（θ は角度の絶対値）です。

## Colab での学習手順

1. `sample/` を zip にして Google Drive に置く（フォルダ直コピーでも可だが、ファイル数が多いと遅い）
2. `autotrim_training.ipynb` を Colab で開き、ランタイムを **GPU** にする
3. 設定セルの `DATA_SOURCE` / `OUTPUT_DIR` を自分の Drive パスに合わせる
4. 上から順に実行する

成果物は `OUTPUT_DIR/日時/` に保存されます。

- `trim_analyzer.keras` / `trim_cropper.keras`
- `trim_*_savedmodel/`（TFLite 変換用）
- `config.json`（学習設定）
- `metrics.json`（検証結果）

### データを増やしたとき

- zip を差し替えて再実行すると、更新を検出して展開し直します
- 学習／検証の分割は撮影シーン名のハッシュで固定されるため、既存データの割り当ては変わりません
- 画像が増えてメモリに収まらなくなると、ディスクキャッシュに自動で切り替わります
- `INIT_FROM_DIR` に前回の保存フォルダを指定すると、同じバックボーン・入力サイズの場合に追加学習できます

## アプリ側の入出力（概要）

| モデル | 入力 | 出力 |
|---|---|---|
| `trim_analyzer` | `[1, A_SIZE, A_SIZE, 3]` float32、RGB 0〜255、縦横比を保って中央配置（余白は黒） | `rotation` / `aspect`(5) / `portrait` |
| `trim_cropper` | `image` `[1, B_SIZE, B_SIZE, 3]`（回転後キャンバスをリサイズ）＋ `condition` `[1, 7]`（比率 one-hot + portrait + `log(目標の幅/高さ ÷ キャンバスの幅/高さ)`） | `box` `[1, 4]` |

詳細と `auto_trim()` の実装はノートブック末尾を参照してください。TFLite 変換処理はリポジトリ外の既存スクリプトを使う想定です。

## ローカルでの動作確認

```bash
export AUTOTRIM_QUICK_TEST=1
export AUTOTRIM_DATA=/path/to/sample
export AUTOTRIM_OUTPUT=/tmp/autotrim_out
# Jupyter で autotrim_training.ipynb を実行（少量データ・1エポック）
```
