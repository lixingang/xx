import gdal
class ReadRaster:
    def __init__(self, path, ):
        self.dataset = gdal.Open(path, gdal.GA_ReadOnly)
        self.rows = self.dataset.RasterYSize  # todo  图像宽度
        self.cols = self.dataset.RasterXSize  # todo  图像长度
        self.bands = self.dataset.RasterCount  # TODO 图像波段数量
        self.proj = self.dataset.GetProjection()  # todo 地图投影信息
        self.geotrans = self.dataset.GetGeoTransform()  # todo 仿射矩阵
    def getRasterInformation(self, nband):
        band = self.dataset.GetRasterBand(nband)  # 获取波段对象
        # data = band.ReadAsArray(0, 0, self.cols, self.rows).astype(numpy.float)  #获取波段信息
        data = band.ReadAsArray(0, 0, self.cols, self.rows)  # 获取波段信息
        return data
    def computedoffset(self):
        data = self.geotrans
        return data
    def computeRows(self):
        return self.rows
    def computeCols(self):
        return self.cols
    def writeRasterInformation(self, data, Savepath, nband):
        driver = self.dataset.GetDriver()
        writeable = driver.Create(Savepath, self.cols, self.rows, self.bands, gdal.GDT_Byte)  # TODO  新建数据集
        writeable.SetGeoTransform(self.geotrans)  # 写入仿射变换参数
        writeable.SetProjection(self.proj)  # 写入投影
        for i in range(nband):
            writeable.GetRasterBand(i + 1).WriteArray(data[i], 0, 0)
            writeable.GetRasterBand(i + 1).SetNoDataValue(0)  # todo 给各波段设置nodata值
            writeable.GetRasterBand(i + 1).FlushCache()  # todo 波段统计量
            print(writeable.GetRasterBand(i + 1).GetStatistics(0, 1))  # todo 计算波段统计量  输出为min\max \Mean\stddev
        return 'success'