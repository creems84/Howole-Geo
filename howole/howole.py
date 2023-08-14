from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from PyQt5 import QtWidgets, uic
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtWidgets import QAction
from .resources import *
from .howole_dialog import HowoleDialog
from PyQt5.QtWidgets import QMessageBox
import os.path
from qgis.core import *
from qgis.gui import QgsMapCanvas, QgsLayerTreeMapCanvasBridge
from qgis.utils import *
from qgis.PyQt import QtGui
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from shapely.geometry import Point
import math
from cmath import *
import numpy as np
import random
import pandas as pd 
from qgis.PyQt.QtCore import QVariant
from qgis.core import QgsWkbTypes
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from scipy.cluster.hierarchy import dendrogram
from random import randrange
from collections import Counter
from scipy.spatial import ConvexHull
import cv2
from cv2 import flip
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QFileInfo
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from itertools import combinations
import mpl_toolkits.axisartist as axisartist

class Howole:
    def __init__(self, iface):             
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'Howole_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = self.tr(u'&Фильтрация линейных объектов')
        self.first_start = None
        
   
    def tr(self, message):
        return QCoreApplication.translate('Howole', message)

    def plot(self, hour, temperature):
        self.widget1.plot(hour, temperature)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        
        icon_path = ':/plugins/howole/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Howole'),
            callback=self.run,
            parent=self.iface.mainWindow())

        self.first_start = True

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Фильтрация линейных объектов'),
                action)
            self.iface.removeToolBarIcon(action)

    def run(self):        
        if self.first_start == True:
            self.first_start = False
            self.dlg = HowoleDialog()
        
                    
        def clear_comboBox():            
            self.dlg.comboBox.clear() 
            curLayers = list(QgsProject.instance().mapLayers().values())
            layerNames = []
            for cL in curLayers:             
                layerNames.append(cL.name())
            self.dlg.comboBox.addItems(layerNames)
        clear_comboBox()

        def chooseIndex():
            sLayerIndex = self.dlg.comboBox.currentIndex() 
            curLayers = list(QgsProject.instance().mapLayers().values()) 
            selectedLayer = curLayers[sLayerIndex]         
            self.iface.setActiveLayer(selectedLayer) 
        self.dlg.comboBox.currentIndexChanged.connect(chooseIndex)

        dict_point_coord = {} #словарь, где ключи id линий,а значения координаты точек пересечений 
        edge = [] #хранит координаты ребер
        intersected_lines = [] #хранит линии, которые пересеклись        
        intersection_steps = [] #хранит шаги, на которых линии пересеклись 
        vertex_numbers=[]#хранит номера вершин
        dict_id_step = {}#словарь, где ключи id линий,которые закончили существование, а значения на каком шаге
        dict_polyline = {} #словарь, где ключи id линий, а значения добавленные линии 
        dict_poly = {} # словарь, где ключи номера полигонов, а значения вершины   
        edge_one=[]#хранит номера ребер
        dict_vertex= {} #словарь, где ключи номера вершин,а значения координаты
        
        # МЕТРИКИ ДЛЯ ОЦЕНКИ КАЧЕСТВА КЛАСТЕРИЗАЦИИ
        # Индекс Данна

        def dunnIndex(line_layer, centroid_layer):
            # Расчет расстояний между центроидами
            centroid_features = centroid_layer.getFeatures()
            centroid_points = [feature.geometry().asPoint() for feature in centroid_features]
            centroid_distances = pairwise_distances(centroid_points)

            # Расчет диаметра кластера (максимальное расстояние между центроидами)
            cluster_diameter = centroid_distances.max()/2  

            # Расчет минимального расстояния между центроидами разных кластеров
            min_intercluster_distance = centroid_distances[centroid_distances > 0].min()

            # Расчет индекса Данна
            dunn_index = min_intercluster_distance / cluster_diameter
            return dunn_index

        # Индекс Инерции
        def inertia(line_layer, centroid_layer1):
            # Получаем все объекты (линии) из слоя "Main_Lines"
            features = line_layer.getFeatures()

            line_points = []
            centroid_points = []

            # Собираем точки линий и центроиды в отдельные списки
            for feature in features:
                line_geometry = feature.geometry()
                if line_geometry.type() == QgsWkbTypes.LineGeometry:
                    line_points.extend(line_geometry.asPolyline())

                # Используем запрос с фильтром по идентификатору объекта
                centroid_request = QgsFeatureRequest().setFilterFid(feature.id())
                centroid_features = centroid_layer1.getFeatures(centroid_request)

                for centroid_feature in centroid_features:
                    centroid_geometry = centroid_feature.geometry()
                    if centroid_geometry.type() == QgsWkbTypes.PointGeometry:
                        centroid_points.append(centroid_geometry.asPoint())

            # Вычисляем расстояния между каждым объектом и его центроидом
            distances = pairwise_distances(line_points, centroid_points)

            # Вычисляем инерцию как сумму квадратов расстояний
            inertia_value = (distances ** 2).sum() / centroid_layer1.featureCount()  ###

            return inertia_value

        # Функция для вычисления индекса Дэвиса-Болдина
        def daviesBouldinIndex(line_layer, centroid_layer):
            # Получаем все объекты (линии) из слоя "Main_Lines"
            features = line_layer.getFeatures()

            line_points = []
            centroid_points = []

            # Собираем точки линий и центроиды в отдельные списки
            for feature in features:
                line_geometry = feature.geometry()
                if line_geometry.type() == QgsWkbTypes.LineGeometry:
                    line_points.extend(line_geometry.asPolyline())

                # Используем запрос с фильтром по идентификатору объекта
                centroid_request = QgsFeatureRequest().setFilterFid(feature.id())
                centroid_features = centroid_layer.getFeatures(centroid_request)

                for centroid_feature in centroid_features:
                    centroid_geometry = centroid_feature.geometry()
                    if centroid_geometry.type() == QgsWkbTypes.PointGeometry:
                        centroid_points.append(centroid_geometry.asPoint())

            # Вычисляем расстояния между каждым объектом и его центроидом
            distances = pairwise_distances(line_points, centroid_points)

            # Вычисляем сходство между каждым кластером и наиболее похожим на него кластером
            similarity_matrix = 1 / (1 + distances)
            similarity_max = np.max(similarity_matrix, axis=1)

            # Вычисляем среднее значение сходства
            similarity_avg = np.mean(similarity_matrix, axis=1)

            # Вычисляем индекс Дэвиса-Болдина как сумму средних значений сходства между кластерами
            davies_bouldin_index = np.mean(similarity_avg / similarity_max)

            return davies_bouldin_index

        def calculateIndices(line_layer_name, centroid_layer_name, centroid_layer1_name):
            line_layer = QgsProject.instance().mapLayersByName(line_layer_name)[0] 
            centroid_layer = QgsProject.instance().mapLayersByName(centroid_layer_name)[0] 
            centroid_layer1 = QgsProject.instance().mapLayersByName(centroid_layer1_name)[0] 

            dunn_index = dunnIndex(line_layer, centroid_layer)
            dunn_index = "{:.3f}".format(dunn_index)

            inertia_value = inertia(line_layer, centroid_layer)            
            inertia_value = "{:.3f}".format(inertia_value)

            davies_bouldin_index_value = daviesBouldinIndex(line_layer, centroid_layer)
            davies_bouldin_index_value = "{:.3f}".format(davies_bouldin_index_value)

            self.dlg.lineEdit_12.setText(str(dunn_index))
            self.dlg.lineEdit_13.setText(str(davies_bouldin_index_value))
            self.dlg.lineEdit_14.setText(str(inertia_value))

        self.dlg.pushButton_analiz_7.clicked.connect(lambda: calculateIndices("Main_Lines", "Main_Centroids", "Line_Points"))

        #линейный алгоритм
        
        def defQuery():            
            sLayerIndex = self.dlg.comboBox.currentIndex()#получаем индекс выбранного пункта в comboBox      
            curLayers = list(QgsProject.instance().mapLayers().values())
            layer = curLayers[sLayerIndex] #переменная равна объекту выбранного слоя
            QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry)
            
            if layer.type() == QgsMapLayer.RasterLayer:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите векторный слой")
               return
            if layer.geometryType() == 0:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите линейный слой")
            else:                                    
                n = float(self.dlg.lineEdit.text())# увеличении линии на величину, которую выберет пользователь                             
                layer = iface.activeLayer()
                features = layer.getFeatures()
                global intersection_coords             
                global dict_graph            
                global dict_poly1
                global step
                global intersected_lines
                global intersection_steps
                global vertex_counter
                global vertex_numbers
                global dict_id_step 
                global dict_polyline   
                global polygon_number             
                global dict_poly          
                global edge_number
                global edge_one
                global dict_vertex           
                global lines
                global segment_list 
                global number
                global v
                global w
                global color
                global par 
                global r
                global e
                global edge
                global dict_point_coord
                global x
                global y
                global sh
            
                if n <0:
                    step = 0
                    #polygon_number = 0 #счетчик номеров пoлигонов
                    vertex_counter=0 #счетчик номеров вершин
                    edge_number=0 #счетчик номеров ребер
                    intersected_lines = [] #хранит линии, которые пересеклись        
                    intersection_steps = [] #хранит шаги, на которых линии пересеклись
                    intersection_coords = [] #хранит координаты точек пересечения 
                    vertex_numbers=[]#хранит номера вершин
                    dict_id_step = {}#словарь, где ключи id линий, а значения шаг конца жизненного цикла линии
                    dict_polyline = {} #словарь, где ключи id линий, а значения добавленные линии
                    dict_point_coord = {} #словарь, где ключи id линий,а значения координаты точек пересечений    
                    edge = [] #хранит координаты ребер
                    coord =[]# список хранит координаты полигона
                    dict_poly = {} # словарь, где ключи номера полигонов, а значения вершины    
                    edge_one=[]#хранит номера ребер
                    dict_vertex= {} #словарь, где ключи номера вершин,а значения координаты
                    dict_graph= {} #словарь, где ключи номера ребер,а значения нач и конеч вершин
                    dict_poly1 = defaultdict(list) #словарь, где ключи номер полигона,а значения начало ЖЦ полигона
                lines = {} # словарь, где ключи id линий, а значения координаты начала и конца линии
                segment_list = [] #Список всех линий, с которыми должна пересечься данная линия
                
                #изменение координат линий векторного слоя на определенное значение n
                for feature in features:
                    geom = feature.geometry() 
                    fd=feature.id()
                    print("fd", fd)
                    if QgsWkbTypes.isSingleType(geom.wkbType()):                            
                        pt1 = geom.asPolyline() 
                        if len(pt1)>1:
                            x1, y1 = pt1[0][0], pt1[0][1] 
                            x2, y2 = pt1[1][0], pt1[1][1]
                    else:
                        pt1 = geom.asMultiPolyline()  
                        x1, y1 = pt1[0][0][0], pt1[0][0][1]   
                        x2, y2 = pt1[0][1][0], pt1[0][1][1]
           
                    b = math.sqrt((x1-x2)**2+(y1-y2)**2) #получаем длину исходного отрезка
                
                    if b > 0:                    
                        sina = (y1-y2)/b
                        cosa = (x1-x2)/b
                        dy = sina * n
                        dx = cosa * n
                        x1 += dx #получили  координаты новой длины
                        y1 += dy
                        x2 -= dx
                        y2 -= dy
                        if not dict_point_coord: # если словарь пуст                        
                            geom1 = QgsGeometry.fromPolyline([QgsPoint(x1, y1), QgsPoint(x2, y2)])#перерисуем слой с новыми координатами
                            kl=layer.dataProvider().changeGeometryValues({ fd:geom1 })
                            layer.triggerRepaint()
                            segment_list.append(fd)#добавляем в список id объекта
                            lines[fd]= [x1, y1, x2, y2]
                        else:                        
                            if fd not in dict_point_coord: # если такого ключа в словаре нет
                                geom1 = QgsGeometry.fromPolyline([QgsPoint(x1, y1), QgsPoint(x2, y2)])#перерисуем слой с новыми координатами
                                kl=layer.dataProvider().changeGeometryValues({ fd:geom1 })
                                layer.triggerRepaint()
                                segment_list.append(fd)#добавляем в список id объекта
                                lines[fd]= [x1, y1, x2, y2]
                            else: # если ключ в словаре есть
                                saw= dict_point_coord[fd] #значение координат от номера линии
                                if len(saw) == 1:                                
                                    x1 = x1
                                    y1 = y1
                                    x2 = x2
                                    y2 = y2                                
                                elif len(saw) == 2:                               
                                    x1 = saw[0][0] 
                                    y1 = saw[0][1] 
                                    x2 = saw[1][0]
                                    y2 = saw[1][1]                                
                                else:                                
                                    x1 = saw[0][0] 
                                    y1 = saw[0][1] 
                                    x2 = saw[-1][0]
                                    y2 =saw[-1][1] 
                                geom1 = QgsGeometry.fromPolyline([QgsPoint(x1, y1), QgsPoint(x2, y2)])#перерисуем слой с новыми координатами
                                kl=layer.dataProvider().changeGeometryValues({ fd:geom1 })
                                layer.triggerRepaint()
                                segment_list.append(fd)#добавляем в список id объекта
                                lines[fd]= [x1, y1, x2, y2]                                              
                            geom1 = QgsGeometry.fromPolyline([QgsPoint(x1, y1), QgsPoint(x2, y2)])#перерисуем слой с новыми координатами
                            kl=layer.dataProvider().changeGeometryValues({ feature.id():geom1 })
                            layer.triggerRepaint()
                            segment_list.append(feature.id())#добавляем в список id объекта
                            lines[feature.id()]= [x1, y1, x2, y2]
                layer.triggerRepaint()


                ###    пересечение   
                
                crossing_candidates = []  # хранит списки сравниваемых линий i и j для поиска пересечений
                sh=0
                # Функция для получения ключа по значению в словаре
                def get_key(d, value):
                    for k, v in d.items():
                        if v == value:
                            return k

                # Функция для вычисления расстояния между точками
                def distance(x1, y1, x2, y2):
                    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                for i, j in combinations(lines.keys(), 2):
                    crossing_candidates.append([i, j])
                    x1, y1, x2, y2 = lines[i]
                    x3, y3, x4, y4 = lines[j]

                    # Составляем формулы двух прямых
                    A1 = y2 - y1
                    A2 = y4 - y3
                    B1 = x1 - x2
                    B2 = x3 - x4
                    C1 = x2 * y1 - y2 * x1
                    C2 = x4 * y3 - y4 * x3

                    # Вычисляем углы между линиями
                    row = abs(B1 * B2 + A1 * A2)
                    r2 = math.sqrt(math.pow(B1, 2) + math.pow(A1, 2))
                    r3 = math.sqrt(math.pow(B2, 2) + math.pow(A2, 2))

                    # Вычисляем координаты точки пересечения
                    if A1 * B2 - A2 * B1 != 0:
                        y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
                        x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
                    else:
                        y, x = 0, 0

                    d1 = distance(x1, y1, x, y)
                    d2 = distance(x, y, x2, y2)
                    d3 = distance(x3, y3, x, y)
                    d4 = distance(x, y, x4, y4)
                    len1 = distance(x1, y1, x2, y2)  # длина линии1
                    len2 = distance(x3, y3, x4, y4)  # длина линии 2
                    eps = 0.0000001
                    val1 = len1 - (d1 + d2)
                    val2 = len2 - (d3 + d4)

                    if abs(val1) < eps and abs(val2) < eps:
                        # Если точка пересечения найдена
                        if [i, j] not in intersected_lines:
                            intersected_lines.append([i, j])  # добавляем в список линии, которые пересеклись
                            intersection_steps.append(step)  # хранит шаги, на которых линии пересеклись
                            intersection_coords.append([x, y])  # хранит координаты точек пересечения
                            dict_point_coord[i] = dict_point_coord.get(i, []) + [[x, y]]
                            dict_point_coord[j] = dict_point_coord.get(j, []) + [[x, y]]
                            vertex_numbers.append(vertex_counter)  # хранит номера вершин
                            dict_vertex[vertex_counter] = [x, y]
                            vertex_counter += 1  
                            
                            #Проверяем были ли линии пересечены другими линиями до текущего шага 
                            N = [val for sublist in intersected_lines for val in sublist]                            
                            if i in N[:-2] and j in N[:-2]:  # если обе линии пересекались ранее
                                for key, value in dict_polyline.items():
                                    if i in value and j in value:
                                        min_value = min(value)
                                        dict_id_step[min_value] = step                                
                                for s in list(dict_polyline.values()):
                                    if i in s:  # найти линии, которые входят в полилинии
                                        l = get_key(dict_polyline, s)  # получаем ключ при значении s
                                        dict_polyline[l] = dict_polyline.get(l, []) + [i, j]
                            elif i in N[:-2]:  # если i пересекалась ранее                                
                                dict_id_step[j] = step
                                for s in list(dict_polyline.values()):
                                    if i in s:  # найти линии, которые входят в полилинии
                                        l = get_key(dict_polyline, s)  # получаем ключ при значении s
                                        dict_polyline[l] = dict_polyline.get(l, []) + [i, j]
                            elif j in N[:-2]:  # если j пересекалась ранее                                
                                dict_id_step[i] = step
                                for s in list(dict_polyline.values()):
                                    if j in s:  # найти линии, которые входят в полилинии
                                        l = get_key(dict_polyline, s)  # получаем ключ при значении s
                                        dict_polyline[l] = dict_polyline.get(l, []) + [i, j]
                            elif i and j not in N[:-2]:  # если обе линии не пересекались ранее                                
                                dict_id_step[j] = step
                                dict_polyline[i] = [i, j]
                            elif len(N) < 3:                                
                                dict_id_step[j] = step
                                dict_polyline[i] = [i, j]

                        sh += 1


                #поиск ребер
                for s in list(dict_point_coord.values()): 
                    if len(s)<2:
                        # Если меньше 2-х точек, нет рёбер
                        continue
                    elif len(s)==2:
                        # Если есть 2 точки, это одно ребро                        
                        if s not in edge:
                            edge.append(s)
                            edge_number += 1
                            edge_one.append(edge_number)    
                            st_v = get_key(dict_vertex, s[0])
                            en_v = get_key(dict_vertex, s[1])
                            dict_graph[edge_number]=[st_v,en_v]                         
                    else:
                        # Если больше 2-х точек, это несколько ребер
                        s.sort()                        
                        for first, next_point in zip(s, s[1:]):
                            fg = [first, next_point]
                            if fg not in edge:
                                edge.append(fg)
                                edge_number += 1
                                st_v = get_key(dict_vertex, first)
                                en_v = get_key(dict_vertex, next_point)
                                dict_graph[edge_number] = [st_v, en_v]
                
                num_edge = len(edge) # подсчитываем количество ребер
                num_point = len(intersection_coords) # подсчитываем количество вершин
           
                   
                #поиск полигонов                                                
            
                N = 100
                graph = [[] for i in range(N)]
                cycles = [[] for i in range(N)]

                #алгоритм обхода графа использует рекурсивную функцию, на вход принимает текущую вершину w
                #предыдущую вершину p, цвета вершин color, родительские вершины par
                def graph_cycle(w, p, color: list, par: list):
                    global number                    
                    if color[w] == 2:
                        return
                    if color[w] == 1:
                        v = []
                        cur = p
                        v.append(cur)
                        while cur != w:
                            cur = par[cur]
                            v.append(cur)
                        cycles[number] = v
                        number += 1
        
                        return
                    par[w] = p
                    color[w] = 1
                    for v in graph[w]:
                        if v == par[w]:
                            continue
                        graph_cycle(v, w, color, par)
                    color[w] = 2

                def addEdge(w, v):
                    graph[w].append(v)
                    graph[v].append(w)

                def processConnectedComponent(component_number):
                    # Обработка компоненты связности
                    bi = 0
                    sdin = []
                    for i in range(0, number):
                        for x in cycles[i]:
                            sdin.append(x)
                            dict_poly[bi] = sdin
                        dict_poly1[bi].append(step)
                        sdin = list(sdin)
                        bi += 1
    
                # Процесс обработки несвязных компонент
                component_number = 0  # Счётчик для компонент связности

                for key in dict_graph:
                    r = dict_graph[key][0]
                    e = dict_graph[key][1]
                    addEdge(r, e)

                color = [0] * N
                par = [0] * N
                number = 1

                # Цикл по вершинам графа
                for vertex in range(1, len(color)):
                    if color[vertex] == 0:
                        # Запускаем обход графа, начиная с найденной вершины
                        graph_cycle(vertex, 0, color, par)

                        # Обработка компоненты связности
                        processConnectedComponent(component_number)
                        component_number += 1


                STP = []
                for hry in list(dict_poly1.values()):
                    STP.append(hry[0])
                STP = STP[1:] 
                if  n > 0:                     
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                    # Жизненный цикл линий график
                    x = dict_id_step.keys()
                    y = dict_id_step.values()                    
                    z = 0
                    ax1.hlines(x, y, z, label='Линии ' + str(x))
                    ax1.set_xlabel(r'$Шаги$', size=12)
                    ax1.set_ylabel(r'$Линии$', size=12)
                    ax1.set_title('Жизненный цикл линий', size=14)
                    ax1.grid()

                    if len(STP) > 0:
                        # Жизненный цикл полигонов график
                        x=dict_poly.keys()
                        y =STP
                        z = step                     
                        ax2.hlines(x, y, z, label='Полигон ' + str(x), color=(0, 0, 0))  
                        ax2.set_xlabel(r'$Шаги$', size=12)
                        ax2.set_ylabel(r'$Полигоны$', size=12)
                        ax2.set_title('Жизненный цикл полигонов', size=14)
                        ax2.grid()

                        plt.subplots_adjust(wspace=0.4)  # Увеличим пространство между графиками по горизонтали
                    plt.show()

                step+=1 
           
        self.dlg.pushButton_analiz.clicked.connect(defQuery)
        
        def defGraph(): 
            n2 = int(self.dlg.lineEdit_2.text())
            for i in range(0, n2):
                defQuery()                         
            
        self.dlg.pushButton_analiz_2.clicked.connect(defGraph)
        
        def k_means():             
            k_num = int(self.dlg.lineEdit_5.text())  
            points = []
            layer = iface.activeLayer()            
            sLayerIndex = self.dlg.comboBox.currentIndex()            
            curLayers = list(QgsProject.instance().mapLayers().values())
            selectedLayer = curLayers[sLayerIndex]     
            name_sloy_1 = selectedLayer.name()
            layerName1 = name_sloy_1
            if selectedLayer.type() == QgsMapLayer.RasterLayer:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите векторный слой")
               return
            if selectedLayer.geometryType() == 0:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите линейный слой")
            else:                   
                layer1 = QgsProject.instance().mapLayersByName(layerName1)[0]
                epsg = layer1.crs().postgisSrid()
                uri1 = "Point?crs=epsg:" + str(epsg) + "&field=id:integer""&index=yes"
                mem_layer1 = QgsVectorLayer(uri1, 'K-means', 'memory')
                agglomerative_provider = mem_layer1.dataProvider()
                agglomerative_provider.addAttributes([QgsField("id", QVariant.Int),
                        QgsField("cluster_id",  QVariant.Int),
                        QgsField("cluster_SIZE", QVariant.Int)])
                mem_layer1.updateFields() 
                for feature in layer1.getFeatures():    
                    geom = feature.geometry()
                    id= feature.id() 
                    if QgsWkbTypes.isSingleType(geom.wkbType()):
                        pt1 = geom.asPolyline() 
                        if len(pt1)>1:
                            x=pt1[0][0]
                            y=pt1[0][1]  
                            x1=pt1[1][0]
                            y1=pt1[1][1] 
                    else:
                        pt1 = geom.asMultiPolyline()                   
                        x=pt1[0][0][0]
                        y=pt1[0][0][1]   
                        x1=pt1[0][1][0]
                        y1=pt1[0][1][1]                
                    points.append([x,y])
                    points.append([x1,y1])
                    #Добавляем новый объект и назначаем геометрию
                    feat1 = QgsFeature()
                    feat2 = QgsFeature()
                    if QgsWkbTypes.isSingleType(geom.wkbType()):
                        if len(pt1)>1:
                            feat1.setGeometry(QgsGeometry.fromPointXY(pt1[0]))  
                            feat2.setGeometry(QgsGeometry.fromPointXY(pt1[1])) 
                    else:
                        feat1.setGeometry(QgsGeometry.fromPointXY(pt1[0][0]))
                        feat2.setGeometry(QgsGeometry.fromPointXY(pt1[0][1]))     
                    feat1.setAttributes([id])
                    feat2.setAttributes([id])
                    agglomerative_provider.addFeatures([feat1]) 
                    agglomerative_provider.addFeatures([feat2])                           
           
                X = np.array(points)
                col=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(k_num)]
                km = KMeans(n_clusters=k_num).fit(X)
                y_l = km.labels_
                centroids = km.cluster_centers_ 
                                   
                a=Counter(y_l)            
                for (f, b) in zip(mem_layer1.getFeatures(), km.labels_):                
                    mem_layer1.dataProvider().changeAttributeValues({f.id() : {1: int(b), 2: int(a[b])}}) 
            
                #Слой центроидов
                layer_cent = QgsVectorLayer('Point?crs=epsg:4326', 'Point_centroid' , 'memory')
                agglomerative_provider1 = layer_cent.dataProvider()                          
                agglomerative_provider1.addAttributes([QgsField("num_cl_km",  QVariant.Int),
                    QgsField("x",  QVariant.Double),
                                    QgsField("y", QVariant.Double)])
                layer_cent.updateFields()
                      
                for j, s in enumerate(centroids):  # j - номер центроида, s - координаты   
                    feat333 = QgsFeature()   
                    points11 = QgsPointXY(centroids[j][0], centroids[j][1])
                    feat333.setGeometry(QgsGeometry.fromPointXY(points11))
                    feat333.setAttributes([int(j),float(centroids[j][0]), float(centroids[j][1])])
                    agglomerative_provider1.addFeatures([feat333])
                layer.updateExtents()
                QgsProject.instance().addMapLayers([layer_cent])           
                QgsProject.instance().addMapLayers([layer])
                QgsProject.instance().addMapLayer(mem_layer1)                                       
            
                #Перерисуем градиент по кластерам            
                fni = mem_layer1.fields().indexFromName('cluster_id')
                unique_values = mem_layer1.uniqueValues(fni)            
                categories = []
                for unique_value in unique_values:
                    # Инициализируем символ по умолчанию для этого типа геометрии
                    symbol = QgsSymbol.defaultSymbol(mem_layer1.geometryType())
                    # Настройка слоя символов
                    layer_style = {}
                    layer_style['color'] = '%d, %d, %d' % (randrange(0, 256), randrange(0, 256), randrange(0, 256))
                    layer_style['outline'] = '#000000'
                    symbol_layer = QgsSimpleFillSymbolLayer.create(layer_style)                
                    if symbol_layer is not None:
                        symbol.changeSymbolLayer(0, symbol_layer)                
                    category = QgsRendererCategory(unique_value, symbol, str(unique_value))                
                    categories.append(category)            
                renderer = QgsCategorizedSymbolRenderer('cluster_id', categories)            
                if renderer is not None:
                    mem_layer1.setRenderer(renderer)
                mem_layer1.triggerRepaint()
            
                #Создаем линейный слой для отрисовки каждого кластера
                layerMVO = QgsVectorLayer('LineString?crs=epsg:4326', 'MVO' , 'memory') 
                provMVO = layerMVO.dataProvider()
                for i in range(k_num):
                    points = X[y_l == i]                
                    if len(points)>2:
                        hull = ConvexHull(points)                
                        vert = np.append(hull.vertices, hull.vertices[0])                 
                        i=0         
                        l=len(vert)
                        for j, s in enumerate(vert):
                            i=i+1
                            if i < l:   
                                start_point = QgsPointXY(points[s] [0], points[s] [1])
                                end_point = QgsPointXY(points[vert[j+1]] [0], points[vert[j+1]] [1])
                                feat = QgsFeature()                   
                                feat.setGeometry(QgsGeometry.fromPolylineXY([start_point, end_point]))
                                provMVO.addFeatures([feat])    
                                layerMVO.updateExtents()
                                names = [layer.name() for layer in QgsProject.instance().mapLayers().values()]            
           
                layerMVO.updateExtents()
                QgsProject.instance().addMapLayers([layerMVO])
            
                def plot_kmeans(X, y_l, centroids, k_num, col):
                    fig = plt.figure(figsize=(6, 8))
                    ax = axisartist.Subplot(fig, 111)
                    fig.add_axes(ax)
                    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
                    ax.axis["left"].set_axisline_style("-|>", size=1.5)
                    ax.axis["top"].set_visible(False)
                    ax.axis["right"].set_visible(False)

                    ax.axis["bottom"].label.set_text(r'$X$') 
                    ax.axis["left"].label.set_text(r'$Y$')  
                    ax.axis["bottom"].label.set_fontsize(18)
                    ax.axis["left"].label.set_fontsize(18)
                    ax.axis["bottom"].label.set_verticalalignment("top")
                    ax.axis["left"].label.set_verticalalignment("top")

                    line_width = 2
                    for i in range(k_num):
                        points1 = X[y_l == i]
                        ax.scatter(points1[:, 0], points1[:, 1], s=100, c=col[i], label=f'Cluster {i + 1}')
                        if len(points1) > 2:
                            hull = ConvexHull(points1)
                            vert = np.append(hull.vertices, hull.vertices[0])
                            ax.plot(points1[vert, 0], points1[vert, 1], c=col[i], linewidth=line_width)
                            ax.fill(points1[vert, 0], points1[vert, 1], c=col[i], alpha=0.2)

                    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='x')
                    ax.legend()
                    ax.axis["bottom"].major_ticklabels.set_fontsize(14)  
                    ax.axis["left"].major_ticklabels.set_fontsize(14)

                    plt.title('K-means. Количество кластеров: %d' % k_num, fontsize=18)  
                    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)                                  
                    plt.show()

                plot_kmeans(X, y_l, centroids, k_num, col)              
                clear_comboBox()
        
        def performAnalysis2():
            k_means()
            project = QgsProject.instance()
            if (
                project.mapLayersByName("Main_Lines")
                and project.mapLayersByName("Point_centroid")
                and project.mapLayersByName("K-means")
            ):
                calculateIndices("Main_Lines", "Point_centroid", "K-means")
        self.dlg.pushButton_analiz_4.clicked.connect(performAnalysis2)

        def DBSCAN_clustering():
            eps = float(self.dlg.lineEdit_4.text())  # Значение eps
            min_samples = int(self.dlg.lineEdit_3.text())  # Минимальное количество точек в окрестности
            points = []  # Список точек                        
            layer = iface.activeLayer()
            sLayerIndex = self.dlg.comboBox.currentIndex()
            curLayers = list(QgsProject.instance().mapLayers().values())
            selectedLayer = curLayers[sLayerIndex]
            if selectedLayer.type() == QgsMapLayer.RasterLayer:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите векторный слой")
               return
            if selectedLayer.geometryType() == 0:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите линейный слой")
            else: 
                name_sloy_1 = selectedLayer.name()                        
                uri = "Point?crs=epsg:" + str(layer.crs().postgisSrid()) + "&field=id:integer""&index=yes"
                mem_layer = QgsVectorLayer(uri, 'DBSCAN', 'memory')
                provider = mem_layer.dataProvider()
                provider.addAttributes([
                    QgsField("id", QVariant.Int),
                    QgsField("cluster_id", QVariant.Int),
                    QgsField("cluster_SIZE", QVariant.Int)
                ])
                mem_layer.updateFields()
            
                for feature in layer.getFeatures():
                    geom = feature.geometry()
                    id = feature.id()
                    if QgsWkbTypes.isSingleType(geom.wkbType()):
                        pt = geom.asPolyline()
                    else:
                        pt = geom.asMultiPolyline()
                    for p in pt:
                        x, y = p[0], p[1]
                        points.append([x, y])                    
                        point_feature = QgsFeature()
                        point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                        point_feature.setAttributes([id])
                        provider.addFeature(point_feature)
                X = np.array(points)  

                # Кластеризация DBSCAN
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = db.labels_
                X = X.reshape(-1, 2)  # Преобразование размерности
                y_l = db.labels_
                # Обрабатываем результаты кластеризации
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                cluster_sizes = Counter(labels)
                for f, cluster_id in zip(mem_layer.getFeatures(), labels):
                    provider.changeAttributeValues({f.id(): {1: int(cluster_id), 2: int(cluster_sizes[cluster_id])}})
                QgsProject.instance().addMapLayer(mem_layer)
                layer.updateExtents()

                #Стиль
                categories = []
                unique_cluster_ids = set([feature["cluster_id"] for feature in mem_layer.getFeatures()])
                for cluster_id in unique_cluster_ids:                
                    symbol = QgsSymbol.defaultSymbol(mem_layer.geometryType())                
                    symbol_layer = QgsSimpleFillSymbolLayer.create({'color': '255,0,0', 'outline_color': '0,0,0', 'outline_width': '0.8'})
                    if symbol_layer is not None:
                        symbol.changeSymbolLayer(0, symbol_layer)                
                    category = QgsRendererCategory(cluster_id, symbol, f'Cluster {cluster_id}')
                    categories.append(category)
                renderer = QgsCategorizedSymbolRenderer('cluster_id', categories)
                mem_layer.setRenderer(renderer)
                mem_layer.triggerRepaint()
           
                def create_cluster_convex_hull_layer(X, labels, cluster_sizes, n_clusters):                
                    layerMVO = QgsVectorLayer('LineString?crs=epsg:4326', 'MVO', 'memory') 
                    provMVO = layerMVO.dataProvider()

                    for i in range(n_clusters):
                        points = X[labels == i]                
                        if len(points) > 2:
                            hull = ConvexHull(points)                
                            vert = np.append(hull.vertices, hull.vertices[0])                 
                            i = 0         
                            l = len(vert)
                            for j, s in enumerate(vert):
                                i = i + 1
                                if i < l:   
                                    start_point = QgsPointXY(points[s][0], points[s][1])
                                    end_point = QgsPointXY(points[vert[j+1]][0], points[vert[j+1]][1])
                                    feat = QgsFeature()                   
                                    feat.setGeometry(QgsGeometry.fromPolylineXY([start_point, end_point]))
                                    provMVO.addFeatures([feat])
                                    layerMVO.updateExtents()
                    return layerMVO
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                cluster_sizes = Counter(labels)

                hull_layer = create_cluster_convex_hull_layer(X, labels, cluster_sizes, n_clusters)

                QgsProject.instance().addMapLayer(hull_layer)
                hull_layer.updateExtents()

                # Минимальная выпуклая оболочка
                col = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(n_clusters)]

                # Строим график

                def plot_dbscan(X, cluster_id, n_clusters, col):
                    fig = plt.figure(figsize=(6, 8))           
                    ax = axisartist.Subplot(fig, 111)           
                    fig.add_axes(ax)            
                    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
                    ax.axis["left"].set_axisline_style("-|>", size=1.5)            
                    ax.axis["top"].set_visible(False)
                    ax.axis["right"].set_visible(False)

                    ax.axis["bottom"].label.set_text(r'$X$')
                    ax.axis["left"].label.set_text(r'$Y$')
                    ax.axis["bottom"].label.set_fontsize(18)
                    ax.axis["left"].label.set_fontsize(18)

                    ax.axis["bottom"].label.set_verticalalignment("top")
                    ax.axis["left"].label.set_verticalalignment("top")
            
                    line_width = 2
                    for cluster_id in range(n_clusters):
                        cluster_points = X[y_l == cluster_id, :]
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c=col[cluster_id], label=f'Cluster {cluster_id + 1}')

                        if len(cluster_points) > 2:
                            hull = ConvexHull(cluster_points)
                            vert = np.append(hull.vertices, hull.vertices[0])
                            ax.plot(cluster_points[vert, 0], cluster_points[vert, 1], c=col[cluster_id], linewidth=line_width)
                            ax.fill(cluster_points[vert, 0], cluster_points[vert, 1], c=col[cluster_id], alpha=0.2)

            
                    ax.legend()
                    ax.axis["bottom"].major_ticklabels.set_fontsize(14)  
                    ax.axis["left"].major_ticklabels.set_fontsize(14)
                    plt.title('DBSCAN.Количество кластеров: %d' % n_clusters, fontsize=18)
                    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9) 
                    plt.show()

                       
                plot_dbscan(X, cluster_id, n_clusters, col)

                clear_comboBox()
                n_noise_ = list(y_l).count(-1)
                self.dlg.textBoxIDSelectedObjects.setText(str(n_clusters)) 
                self.dlg.textBoxIDSelectedObjects2.setText(str(n_noise_))

        def performAnalysis1():
            DBSCAN_clustering()
            project = QgsProject.instance()
            if (
                project.mapLayersByName("Main_Lines")
                and project.mapLayersByName("Main_Centroids")
                and project.mapLayersByName("DBSCAN")
            ):
                calculateIndices("Main_Lines", "Main_Centroids", "DBSCAN")

        self.dlg.pushButton_analiz_3.clicked.connect(performAnalysis1)

        def Agglomer():
            def chooseIndex():            
                sLayerIndex = self.dlg.comboBox.currentIndex() 
                selectedLayer = curLayers[sLayerIndex]            
                self.iface.setActiveLayer(selectedLayer) 
            self.dlg.comboBox.currentIndexChanged.connect(chooseIndex)
            
            clust_num = int(self.dlg.lineEdit_6.text())
            linkage = self.dlg.comboBox_2.currentText()            
            svo=[]
            layer = iface.activeLayer()            
            sLayerIndex = self.dlg.comboBox.currentIndex()            
            curLayers = list(QgsProject.instance().mapLayers().values())
            selectedLayer = curLayers[sLayerIndex]   
            if selectedLayer.type() == QgsMapLayer.RasterLayer:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите векторный слой")
               return
            if selectedLayer.geometryType() == 0:
               QMessageBox.warning(self.dlg, "Ошибка", "Выберите линейный слой")
            else: 
                name_sloy_1 = selectedLayer.name()
                layerName1 = name_sloy_1
                layer1 = QgsProject.instance().mapLayersByName(layerName1)[0]
                epsg = layer1.crs().postgisSrid()# получаем проекцию
                uri1 = "Point?crs=epsg:" + str(epsg) + "&field=id:integer""&index=yes"
                agglomerative_layer = QgsVectorLayer(uri1, 'Aglomerative', 'memory')
                agglomerative_provider = agglomerative_layer.dataProvider()
                agglomerative_provider.addAttributes([QgsField("id", QVariant.Int),
                        QgsField("cluster_id",  QVariant.Int),
                        QgsField("cluster_SIZE", QVariant.Int)])
                agglomerative_layer.updateFields() 
                for feature in layer1.getFeatures():
                    geom = feature.geometry()
                    id= feature.id()                
                    if QgsWkbTypes.isSingleType(geom.wkbType()):
                        pt1 = geom.asPolyline()
                        if len(pt1)>1:
                            x=pt1[0][0] 
                            y=pt1[0][1]   
                            x1=pt1[1][0] 
                            y1=pt1[1][1] 
                    else:
                        pt1 = geom.asMultiPolyline()                   
                        x=pt1[0][0][0]
                        y=pt1[0][0][1]   
                        x1=pt1[0][1][0]
                        y1=pt1[0][1][1] 
                    svo.append([x,y])
                    svo.append([x1,y1])
                
                    #Добавляем новый объект и назначаем геометрию
                    feat1 = QgsFeature()
                    feat2 = QgsFeature()
                                
                    if QgsWkbTypes.isSingleType(geom.wkbType()):
                        if len(pt1)>1:
                            feat1.setGeometry(QgsGeometry.fromPointXY(pt1[0]))  
                            feat2.setGeometry(QgsGeometry.fromPointXY(pt1[1]))  
                    else:
                        feat1.setGeometry(QgsGeometry.fromPointXY(pt1[0][0])) 
                        feat2.setGeometry(QgsGeometry.fromPointXY(pt1[0][1]))

                    feat1.setAttributes([id])
                    feat2.setAttributes([id])
                    agglomerative_provider.addFeatures([feat1]) 
                    agglomerative_provider.addFeatures([feat2])
                               
                X = np.array(svo)
                col=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(clust_num)]
                clustering = AgglomerativeClustering(linkage=linkage, n_clusters=clust_num).fit(X)             
                labels = clustering.labels_  
                n_clusters_ = len(set(labels))     
            
                print('ИТОГО кластеров: %d' % n_clusters_)             
                a=Counter(labels)            
                for (f, b) in zip(agglomerative_layer.getFeatures(), clustering.labels_):                
                    agglomerative_layer.dataProvider().changeAttributeValues({f.id() : {1: int(b), 2: int(a[b])}})
                layer.updateExtents()
                QgsProject.instance().addMapLayers([layer])
                QgsProject.instance().addMapLayer(agglomerative_layer)                                       
                        
                #Задаем градиент           
                fni = agglomerative_layer.fields().indexFromName('cluster_id')
                unique_values = agglomerative_layer.uniqueValues(fni)            
                categories = []
                for unique_value in unique_values:               
                    symbol = QgsSymbol.defaultSymbol(agglomerative_layer.geometryType())               
                    layer_style = {}
                    layer_style['color'] = '%d, %d, %d' % (randrange(0, 256), randrange(0, 256), randrange(0, 256))
                    layer_style['outline'] = '#000000'
                    symbol_layer = QgsSimpleFillSymbolLayer.create(layer_style)                
                    if symbol_layer is not None:
                        symbol.changeSymbolLayer(0, symbol_layer)               
                    category = QgsRendererCategory(unique_value, symbol, str(unique_value))                
                    categories.append(category)            
                renderer = QgsCategorizedSymbolRenderer('cluster_id', categories)            
                if renderer is not None:
                    agglomerative_layer.setRenderer(renderer)
                agglomerative_layer.triggerRepaint()
            
                #Создаем линейный слой для отрисовки каждого кластера
                layerMVO_Al = QgsVectorLayer('LineString?crs=epsg:4326', 'MVO_Agl' , 'memory') 
                provMVO_Al = layerMVO_Al.dataProvider()
            
                for i in range(clust_num):
                    points = X[labels == i]                
                    if len(points)>2:
                        hull = ConvexHull(points)                
                        vert = np.append(hull.vertices, hull.vertices[0])                 
                        i=0         
                        l=len(vert)
                        for j, s in enumerate(vert):
                            i=i+1
                            if i < l:   
                                start_point = QgsPointXY(points[s] [0], points[s] [1])
                                end_point = QgsPointXY(points[vert[j+1]] [0], points[vert[j+1]] [1])
                                feat = QgsFeature()                   
                                feat.setGeometry(QgsGeometry.fromPolylineXY([start_point, end_point]))
                                provMVO_Al.addFeatures([feat])    
                                layerMVO_Al.updateExtents()
                                names = [layer.name() for layer in QgsProject.instance().mapLayers().values()]            
           
                provMVO_Al.updateExtents()
                QgsProject.instance().addMapLayers([layerMVO_Al])
            
                # Создаем график
                def plot_agglomerative_clustering (X, labels, clust_num, col):
                    fig = plt.figure(figsize=(6, 8))
                    ax = axisartist.Subplot(fig, 111)
                    fig.add_axes(ax)
                    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
                    ax.axis["left"].set_axisline_style("-|>", size=1.5)
                    ax.axis["top"].set_visible(False)
                    ax.axis["right"].set_visible(False)

                    ax.axis["bottom"].label.set_text(r'$X$') 
                    ax.axis["left"].label.set_text(r'$Y$')  
                    ax.axis["bottom"].label.set_fontsize(18)
                    ax.axis["left"].label.set_fontsize(18)
                        
                    ax.axis["bottom"].label.set_verticalalignment("top")
                    ax.axis["left"].label.set_verticalalignment("top")
                    line_width = 2
                    for i in range(clust_num):    
                        points1 = X[labels == i]                   
                        ax.scatter(points1[:, 0], points1[:, 1], s=100, c=col[i], label=f'Cluster {i + 1}')
                        if len(points)>2:
                            hull = ConvexHull(points1)
                            vert = np.append(hull.vertices, hull.vertices[0]) 
                            ax.plot(points1[vert, 0], points1[vert, 1], c=col[i], linewidth=line_width)
                            ax.fill(points1[vert, 0], points1[vert, 1], c=col[i], alpha=0.2)
                    ax.legend()                        
                    ax.axis["bottom"].major_ticklabels.set_fontsize(14)  
                    ax.axis["left"].major_ticklabels.set_fontsize(14)

                    plt.title('Агломеративная кластеризация. \n Количество кластеров: %d' % n_clusters_, fontsize=18)           
                    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)  
                    plt.show()
                plot_agglomerative_clustering (X, labels, clust_num, col)
                centroids = []
                for i in range(clust_num):
                    points = X[labels == i]
                    centroid = np.mean(points, axis=0)
                    centroids.append(centroid)
                centroid_layer = QgsVectorLayer('Point?crs=epsg:' + str(epsg), 'Centroids', 'memory')
                centroid_provider = centroid_layer.dataProvider()
                centroid_fields = QgsFields()
                centroid_fields.append(QgsField('cluster_id', QVariant.Int))
                centroid_layer.startEditing()

                for i, centroid in enumerate(centroids):
                    feature = QgsFeature()
                    feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(centroid[0], centroid[1])))
                    feature.setFields(centroid_fields)
                    feature['cluster_id'] = i
                    centroid_provider.addFeature(feature)

                centroid_layer.updateExtents()
                centroid_layer.commitChanges()
                QgsProject.instance().addMapLayer(centroid_layer)
                clear_comboBox()

        def performAnalysis3():
            Agglomer()
            project = QgsProject.instance()
            if (
                project.mapLayersByName("Main_Lines")
                and project.mapLayersByName("Centroids")
                and project.mapLayersByName("Aglomerative")
            ):
                calculateIndices("Main_Lines", "Centroids", "Aglomerative")                
        self.dlg.pushButton_analiz_5.clicked.connect(performAnalysis3)

        def dendro(): 
            p_num = int(self.dlg.lineEdit_6.text())
            DenX=[]
            DenY=[]
            layer = iface.activeLayer() 
            if layer.name() == "Main_Lines":
                features = layer.getFeatures()
                for feature in features:
                    geom = [v for v in feature.geometry().constGet().vertices()]
                    x = geom[0].x()    
                    y = geom[0].y()                
                    x1 = geom[1].x()    
                    y1 = geom[1].y()
                    DenX.append(x)
                    DenY.append(y)
                    DenX.append(x1)
                    DenY.append(y1)                
                data = {'x': DenX,
                        'y': DenY
                       } 
                df = pd.DataFrame(data, columns=['x', 'y'])
                X= df
                def plot_dendrogram(model, **kwargs):  
                    counts = np.zeros(model.children_.shape[0])
                    n_samples = len(model.labels_)
                    for i, merge in enumerate(model.children_):
                        current_count = 0
                        for child_idx in merge:
                            if child_idx < n_samples:
                                current_count += 1 
                            else:
                                current_count += counts[child_idx - n_samples]
                        counts[i] = current_count
                    linkage_matrix = np.column_stack(
                        [model.children_, model.distances_, counts]
                    ).astype(float)                
                    dendrogram(linkage_matrix, **kwargs)   
                model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
                model = model.fit(X)
                print(model.fit(X))
                plt.title("Дендрограмма иерархической кластеризации")
                plot_dendrogram(model, truncate_mode="level", p=p_num)
                plt.xlabel("Количество точек (или индекс точки, если скобки отсутствуют)")
                plt.show()
            else:
                QMessageBox.warning(None, "Ошибка", "Выберите слой 'Main_Lines'")
        
        def performAnalysis4():
            dendro()
            calculateIndices("Main_Lines", "Centroids", "Aglomerative") 
            
        self.dlg.pushButton_analiz_6.clicked.connect(performAnalysis4)
       
        #Кнопка // "ЗАГРУЗИТЬ" //

        def open_image_dialog():
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=None, caption='Открыть изображение', filter="Изображение (*.png *.jpg *.bmp)"
            )
            return filename

        def load_raster_layer(imagePath):
            fi = QFileInfo(imagePath)
            fname = fi.baseName()
            rlayer = iface.addRasterLayer(imagePath, fname)
            stats = rlayer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)

        def populate_combo_box():
            self.dlg.comboBox.clear()
            curLayers = list(QgsProject.instance().mapLayers().values())
            layerNames = [cL.name() for cL in curLayers]
            self.dlg.comboBox.addItems(layerNames)

        def on_click():
            global imagePath
            imagePath = open_image_dialog()
            if imagePath:
                load_raster_layer(imagePath)
                populate_combo_box()
            return imagePath

        self.dlg.SrcOpenButtonClick.clicked.connect(on_click)
                
         #Кнопка // "ВЕКТОРИЗОВАТЬ" //

        def process_image(imagePath):
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            min_area = int(self.dlg.lineEdit_7.text())
            door_step = int(self.dlg.lineEdit_8.text())
            max_door_step = int(self.dlg.lineEdit_9.text())
            filter_min = int(self.dlg.lineEdit_10.text())
            filter_max = int(self.dlg.lineEdit_11.text())
            
            # Применяем фильтры для удаления шума и улучшения контрастности
            blur = cv2.GaussianBlur(gray, (filter_min, filter_max), 0)
            thresh = cv2.threshold(blur, door_step, max_door_step, cv2.THRESH_BINARY)[1]            
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img_with_rectangles = np.copy(img)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img_with_rectangles, [box], 0, (0, 255, 0), 2)

            result_path = 'D:/Qgis1/Test/Result.jpg'
            cv2.imwrite(result_path, img_with_rectangles)
            
            # Обрабатываем контуры и создаем линейный слой в QGIS
            img = cv2.imread('D:/Qgis1/Test/Result.jpg')
            if img is None:
                raise Exception("Не удалось загрузить изображение")
              
            # Создаем новый слой с линиями

            line_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Main_Lines", "memory")
            line_provider = line_layer.dataProvider()

            # Создаем новый слой с центроидами
            centroid_layer = QgsVectorLayer("Point?crs=EPSG:4326", "Main_Centroids", "memory")
            centroid_provider = centroid_layer.dataProvider()

            # Отображаем прямоугольники и создаем центроиды
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area: # минимальная площадь контура для учета
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
      
                    line1 = QgsGeometry.fromPolyline([QgsPoint(box[1][0], box[1][1]), QgsPoint(box[0][0], box[0][1])])
                    line2 = QgsGeometry.fromPolyline([QgsPoint(box[2][0], box[2][1]), QgsPoint(box[1][0], box[1][1])])
                    line3 = QgsGeometry.fromPolyline([QgsPoint(box[3][0], box[3][1]), QgsPoint(box[2][0], box[2][1])])
                    line4 = QgsGeometry.fromPolyline([QgsPoint(box[0][0], box[0][1]), QgsPoint(box[3][0], box[3][1])])  
                                        
                    # Создаем геометрию центроида на основе координат прямоугольника
                    centroid = QgsGeometry.fromPointXY(QgsPointXY(rect[0][0], rect[0][1]))
        
                    #Создаем геометрию для слоя с линиями и добавляем ее
                    line_feature1 = QgsFeature()
                    line_feature1.setGeometry(line1)
                    line_provider.addFeature(line_feature1)

                    line_feature2 = QgsFeature()
                    line_feature2.setGeometry(line2)
                    line_provider.addFeature(line_feature2)

                    line_feature3 = QgsFeature()
                    line_feature3.setGeometry(line3)
                    line_provider.addFeature(line_feature3)

                    line_feature4 = QgsFeature()
                    line_feature4.setGeometry(line4)
                    line_provider.addFeature(line_feature4)

                    # Создаем геометрию для слоя с центроидами и добавляем ее
                    centroid_feature = QgsFeature()
                    centroid_feature.setGeometry(centroid)
                    centroid_provider.addFeature(centroid_feature)

            # Записываем слои в источники данных            
            line_layer.updateExtents()
            centroid_layer.updateExtents()

            # Добавляем слои в QGIS
            QgsProject.instance().addMapLayer(line_layer)
            QgsProject.instance().addMapLayer(centroid_layer)              
            line_layer = QgsProject.instance().mapLayersByName("Main_Lines")[0]

            # Создаем новый слой с точками
            point_layer = QgsVectorLayer("Point?crs=EPSG:4326", "Line_Points", "memory")
            point_provider = point_layer.dataProvider()

            # Получаем все объекты (линии) из слоя "Main_Lines"
            features = line_layer.getFeatures()

            # Обрабатываем каждую линию
            for idx, feature in enumerate(features):                
                line_geometry = feature.geometry()
                if line_geometry.type() == QgsWkbTypes.LineGeometry:
                    line_points = line_geometry.asPolyline()
                    start_point = line_points[0]
                    end_point = line_points[-1]

                    # Вычисляем среднюю точку между начальной и конечной точкой
                    center_point = QgsPointXY((start_point.x() + end_point.x()) / 2, (start_point.y() + end_point.y()) / 2)

                    # Создаем геометрию для средней точки
                    center_point_geometry = QgsGeometry.fromPointXY(center_point)
                    center_point_feature = QgsFeature()
                    center_point_feature.setGeometry(center_point_geometry)

                    # Задаем метку для точки
                    center_point_feature.setAttributes([idx])

                    point_provider.addFeature(center_point_feature)

            # Записываем слой в источник данных
            point_provider.addFeatures([center_point_feature])
            point_layer.updateExtents()

            # Добавляем слой в QGIS
            QgsProject.instance().addMapLayer(point_layer)

            return result_path

        def show_results(imagePath, result_path):
            imagePath = cv2.imread(imagePath)
            img_with_rectangles = cv2.imread(result_path)

            plt.subplot(121), plt.imshow(cv2.cvtColor(imagePath, cv2.COLOR_BGR2RGB))
            plt.title('Исходное изображение'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(cv2.cvtColor(img_with_rectangles, cv2.COLOR_BGR2RGB))
            plt.title('Результат'), plt.xticks([]), plt.yticks([])
            plt.show()
           
        def Canny_sloy(contours):           
            if imagePath:
                resultPath = process_image(imagePath)                
                show_results(imagePath, resultPath)
                                                   
            #Очищаем comboBox и заполняем заново
            clear_comboBox()
            
            # Получаем индекс элемента с именем "Lines" в comboBox
            index = self.dlg.comboBox.findText("Main_Lines")

            # Активируем слой по индексу
            if index >= 0:
                self.dlg.comboBox.setCurrentIndex(index)

            defQuery()
        self.dlg.pushButton_2.clicked.connect(Canny_sloy)

        isrunning=plugins["howole"]
        if (isrunning.dlg):
          if isrunning.dlg.isVisible():
            isrunning.dlg.activateWindow()
            return
        
        #self.dlg.setModal(True)
        self.dlg.show()
        result = self.dlg.exec_()
        
        if result:
            pass


