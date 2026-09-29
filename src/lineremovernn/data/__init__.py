from lineremovernn.data.ai2d import AI2DDataset
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.mathwriting import MathWritingDataset

downloadable_datasets = {
    "iam": IAMDataset(),
    "mathwriting": MathWritingDataset(),
    "ai2d": AI2DDataset(),
}
