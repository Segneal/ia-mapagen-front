import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils_only_person as vis_util
import time
import csv
import multiprocessing
import wx
import datetime
from pynput import keyboard

DATETIME = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = open("output_log_" + DATETIME + ".txt", "w+")
LOG_FILE.close()


#FUNCIONES

#LECTURA DE VARIABLES POR MEDIO DE ARGV
def titulos():
    printLog("Fecha y Hora: " + DATETIME + "\n")
    print("______ _                           _     _       ")
    print("| ___ (_)                         (_)   | |      ")
    print("| |_/ /_  ___ _ ____   _____ _ __  _  __| | ___  ")
    print("| ___ \ |/ _ \ '_ \ \ / / _ \ '_ \| |/ _` |/ _ \ ")
    print("| |_/ / |  __/ | | \ V /  __/ | | | | (_| | (_) |")
    print("\____/|_|\___|_| |_|\_/ \___|_| |_|_|\__,_|\___/ ")
    print("\n")
    print("Este software es de código abierto (open source) y fue desarrollado para analizar un archivo de video y detectar a todas las personas que se encuentran en cada frame del video.\n")
    print("El programa genera un archivo .CSV con la ubicación de cada persona detectada en cada frame analizado y tambien se puede guardar un video de salida en el cual se visualizan todas las personas detectadas en cada frame.\n")
    print("Al finalizar el procesamiento también se podrá generar un mapa de calor para visualizar claramente las áreas de concentración de las personas en el video analizado. \n")
    print("\n")
    print("Esta aplicación fue desarrollada en colaboración con: \n")
    print(" _   _ _   _  _                         ___  _                              _        ")
    print("| | | | \ | || |              ___      / _ \| |                            | |       ")
    print("| | | |  \| || |     __ _    ( _ )    / /_\ \ |__  _ __ ___ _ __ ___   __ _| |_ ___  ")
    print("| | | | . ` || |    / _` |   / _ \/\  |  _  | '_ \| '__/ _ \ '_ ` _ \ / _` | __/ _ \ ")
    print("| |_| | |\  || |___| (_| |  | (_>  <  | | | | |_) | | |  __/ | | | | | (_| | ||  __/ ")
    print(" \___/\_| \_/\_____/\__,_|   \___/\/  \_| |_/_.__/|_|  \___|_| |_| |_|\__,_|\__\___| ")
    print("                                                      2019 por Ezequiel Scordamaglia ")
    print("\n")


#LEE VALORES DEL SYS ARG Y ASIGNA A LOS PARAMETROS INTERNOS
VIDEO_INPUT_PATH = sys.argv[1]
SAVE_OUTPUT = sys.argv[2]
VIDEO_OUTPUT_PATH = sys.argv[3]
CSV_OUTPUT_PATH = sys.argv[4]
PATH_FROZEN = sys.argv[5]
PATH_TO_LABELS = sys.argv[6]
PORCENTAJE_COINCIDENCIA = sys.argv[7]
FRAME_GROUP = sys.argv[8]
VISUALIZAR_VIDEO = sys.argv[9]
MAP_OUTPUT = sys.argv[10]
NUM_CLASSES = sys.argv[11]


#MUESTRA AL USUARIO LOS VALORES DE LOS PARAMETROS
def printParameterValues():
    printLog("Ingreso de variables por parte del usuario..\n")
    printLog("VIDEO DE ENTRADA:\n")
    printLog(VIDEO_INPUT_PATH + "\n")

    printLog("GUARDAR OUTPUT:\n")
    printLog(str(SAVE_OUTPUT) + "\n")

    if (SAVE_OUTPUT):
        printLog("VIDEO DE SALIDA:\n")
        printLog(VIDEO_OUTPUT_PATH + "\n")

    printLog("ARCHIVO CSV DE SALIDA: \n")
    printLog(CSV_OUTPUT_PATH + "\n")

    printLog("ARCHIVO frozen_inference_graph.pb:\n")
    printLog(PATH_FROZEN + "\n")

    printLog("ARCHIVO mscoco_label_map.pbtxt: \n")
    printLog(PATH_TO_LABELS + "\n")

    printLog("PORCENTAJE DE COINCIDENCIA MINIMO: \n")
    printLog(str(PORCENTAJE_COINCIDENCIA) + "\n")

    printLog("GRUPO DE FRAMES:\n")
    printLog(str(FRAME_GROUP) + "\n")

    printLog("VISUALIZAR EL VIDEO MIENTRAS SE PROCESA:\n")
    printLog(str(VISUALIZAR_VIDEO) + "\n")

    printLog("GENERAR MAPA DE CALOR: \n")
    printLog(str(MAP_OUTPUT) + "\n")

    ###########################################################################
    ###########################################################################

def printLog(text):
    LOG_FILE = open("output_log_" + DATETIME + ".txt", "a")
    DATETIME_LOG = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE.write(DATETIME_LOG + " - " + text)
    LOG_FILE.close()
    print(text)

def bar(total, current, length=10, filler="#", space=" ", border="[]", suffix="", elapsed_time=0):
    
    prefix = str(round(((current/total) * 100), 1)) + "% "
    time_remaining = ((elapsed_time * 100) // ((current/total) * 100)) - elapsed_time
    suffix = "(Transcurrido: " + str(datetime.timedelta(seconds=int(elapsed_time))) + " / Restante: " + str(datetime.timedelta(seconds=int(time_remaining))) + ")"
    printLog(prefix + border[0] + (filler * int(current / total * length) +
                                      (space * (length - int(current / total * length)))) + border[1] + suffix + "\r\n")

def get_path(wildcard, text):
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, text, wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = "CANCELAR"
    dialog.Destroy()
    return path

def get_path_save(wildcard, text):
    style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
    dialog = wx.FileDialog(None, text, wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = "CANCELAR"
    dialog.Destroy()
    return path

def get_value(default, text):
    dlg = wx.TextEntryDialog(None, text, 'Ingrese un valor', default)
    if dlg.ShowModal() == wx.ID_OK:
        retorno = dlg.GetValue()
    else:
        retorno = default
    dlg.Destroy()
    return retorno

def get_yes_no(text):
    resultado = False
    dlg = wx.MessageDialog(None, text,'Atención',wx.YES_NO | wx.ICON_QUESTION)
    result = dlg.ShowModal()
    if result == wx.ID_YES:
        resultado = True
    else:
        resultado = False
    return resultado

def kde_quartic(d,h):
    dn=d/h
    P=(15/16)*(1-dn**2)**2
    return P

def on_press(key):
    if key == keyboard.Key.f12:
        return False



###################################################################################################################
###################################################################################################################

#PROGRAMA PRINCIPAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

titulos()
printParameterValues()


printLog("Iniciando programa...\n")

app = wx.App()
app.MainLoop()


printLog("Recuperando información del video de entrada... \n")

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

if cap.isOpened(): 
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if (MAP_OUTPUT):

    MIN = WIDTH//15
    DEF = WIDTH//6
    MAX = WIDTH//3

    CANTIDAD = int(get_value(str(DEF), 'Ingrese cantidad de cuadrados de la grilla (Entre ' + str(MIN) + ' y ' + str(MAX) + '):'))
    printLog("CANTIDAD DE CUADRADOS: \n")
    printLog(str(CANTIDAD) + "\n")

    GRILLA = WIDTH//CANTIDAD

    H_MIN = int(0.08 * WIDTH)
    H_DEF = int(0.14 * WIDTH)
    H_MAX = int(0.16 * WIDTH)

    H = int(get_value(str(H_DEF), 'Ingrese el radio (H) (Entre ' + str(H_MIN) + ' y ' + str(H_MAX) + '):'))
    printLog("H: \n")
    printLog(str(H) + "\n")


if (SAVE_OUTPUT):
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), FPS, (WIDTH,HEIGHT))

#Vacío el archivo CSV
CSV = open(CSV_OUTPUT_PATH, "w")
CSV.truncate()
CSV.close()

frame = 0
n_cpus = multiprocessing.cpu_count()
config = tf.ConfigProto(inter_op_parallelism_threads=n_cpus)

printLog("Utilizando " + str(n_cpus) + " nucleos para procesar\n")

printLog("Cargando la Red Neuronal... \n")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_FROZEN, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

start_time = time.time()

with keyboard.Listener(on_press=on_press) as listener:
    with detection_graph.as_default():
        with tf.Session(config=config, graph=detection_graph) as sess:
            try:
                printLog("Leyendo imagenes del video... \n")

                printLog("***************************ATENCIÓN***************************\n")
                printLog("Puede detener el programa presionando la tecla F12 \n")
                printLog("**************************************************************\n")
                
                start_time = time.time()

                while True:
                    ret, image_np=cap.read()

                    if ret==True:

                        frame = frame + 1

                        if frame == 1:
                            image_out = image_np #Guardo el primer frame para generar el mapa 

                        if frame%FRAME_GROUP == 0:
                            
                            #image_np = cv2.resize(image_np, (WIDTH,HEIGHT))

                            # Definite input and output Tensors for detection_graph
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Each box represents a part of the image where a particular object was detected.
                            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Each score represent how level of confidence for each of the objects.
                            # Score is shown on the result image, together with the class label.
                            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            # Actual detection.
                            (boxes, scores, classes, num) = sess.run(
                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                            #poner punto bajo cada box
                            vis_util.draw_points_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=1,
                            min_score_thresh=PORCENTAJE_COINCIDENCIA,
                            radius = 2,
                            color = 'red')

                            #recuperar coordenadas para guardarlas en archivo
                            coordinates_person = vis_util.return_coordinates_person(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            frame,
                            use_normalized_coordinates=True,
                            line_thickness=1,
                            min_score_thresh=PORCENTAJE_COINCIDENCIA
                            )

                            #Guardo en el csv toda la informacion de las coordenadas
                            with open(CSV_OUTPUT_PATH, 'a+') as csvFile:
                                writer = csv.writer(csvFile, delimiter=',', lineterminator='\r')
                                writer.writerows(coordinates_person)
                            csvFile.close()

                            if (VISUALIZAR_VIDEO or SAVE_OUTPUT):
                                # Visualization of the results of a detection.
                                vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=1,
                                    min_score_thresh=PORCENTAJE_COINCIDENCIA)

                            if (SAVE_OUTPUT):
                                out.write(image_np)

                            if (VISUALIZAR_VIDEO):
                                cv2.imshow('Imagen de camara ',image_np)

                            if (VISUALIZAR_VIDEO or SAVE_OUTPUT):
                                if cv2.waitKey(20) & 0xFF==ord('q'):
                                    elapsed_time = time.time() - start_time
                                    printLog("Proceso detenido por el usuario: Tiempo Transcurrido: " + str(datetime.timedelta(seconds=int(elapsed_time))) + "\n")
                                    break
                                    cap.release()
                                    if (SAVE_OUTPUT):
                                        out.release()                            
                                    cv2.destroyAllWindows()

                            elapsed_time = time.time() - start_time
                            bar(TOTAL_FRAMES, frame, length=50, elapsed_time=elapsed_time)

                            if not listener.running:
                                raise KeyboardInterrupt
                                 
                    else:
                        elapsed_time = time.time() - start_time
                        printLog("Proceso Finalizado. Tiempo Transcurrido: " + str(datetime.timedelta(seconds=int(elapsed_time))) + "\n")
                        break
                    
            except KeyboardInterrupt:
                printLog("Finalización del procesamiento de video por interrupción del usuario \n")
            except Exception:
                printLog("Finalización del procesamiento de video \n")
            finally:

                elapsed_time = time.time() - start_time
                printLog("Tiempo Transcurrido del procesamiento del video: " + str(datetime.timedelta(seconds=int(elapsed_time))) + "\n")
                
                if (SAVE_OUTPUT):
                    out.release()
                cap.release()
                cv2.destroyAllWindows()   

                if (MAP_OUTPUT):
                   
                    printLog("Generando mapa de calor... \n")
                    
                    start_time2 = time.time()

                    x = []
                    y = []

                    with open(CSV_OUTPUT_PATH, 'r') as csvFile:
                        reader = csv.reader(csvFile)
                        for row in reader:
                            x.append(int(row[1]))
                            y.append(int(row[2]))
                    csvFile.close()

                    #DEFINO GRID SIZE Y RADIO(h)
                    grid_size=GRILLA
                    h=H

                    #OBTENGO X,Y MIN Y MAX
                    x_min=0
                    x_max=WIDTH
                    y_min=0
                    y_max=HEIGHT

                    #CONSTRUYO LA GRILLA
                    x_grid=np.arange(x_min-h,x_max+h,grid_size)
                    y_grid=np.arange(y_min-h,y_max+h,grid_size)
                    x_mesh, y_mesh=np.meshgrid(x_grid, y_grid)

                    #ENCUENTRO LOS PUNTOS MEDIOS DE CADA CUADRADO DE LA GRILLA
                    xc=x_mesh+(grid_size/2)
                    yc=y_mesh+(grid_size/2)

                    #PROCESO
                    intensity_list=[]
                    for j in range(len(xc)):
                        intensity_row=[]
                        for k in range(len(xc[0])):
                            kde_value_list=[]
                            for i in range(len(x)):
                                #CALCULATE DISTANCE
                                d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
                                if d<=h:
                                    p=kde_quartic(d,h)
                                else:
                                    p=0
                                kde_value_list.append(p)
                            #SUM ALL INTENSITY VALUE
                            p_total=sum(kde_value_list)
                            intensity_row.append(p_total)
                        intensity_list.append(intensity_row)
                        elapsed_time2 = time.time() - start_time2
                        bar(len(xc), j+1, length=50, elapsed_time=elapsed_time2)

                    #SALIDA
                    plt.imshow(image_out)
                    intensity=np.array(intensity_list)
                    plt.pcolormesh(x_mesh,y_mesh,intensity, alpha=0.5, cmap="jet", antialiased=True, linewidth=0)
                    #plt.plot(x,y,'ro', alpha=0.3) #Todos los puntos sobre la imagen
                    plt.colorbar()
                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)
                    plt.gca().invert_yaxis()
                    #plt.contour(x_mesh,y_mesh,intensity, 8, alpha=0.5, colors='black')#contorno por areas
                    plt.axis('off')

                    #Guardar imagen automaticamente (No funciona bien)
                    ##fig = plt.figure()
                    ##fig.set_figheight(9)
                    ##fig.set_figwidth(16)
                    ##fig.savefig('C:\Proyecto\Output\HeatMaps\HeatMap G' + str(grid_size) + ' H' + str(h) + '.png', dpi=300)
                    
                    elapsed_time2 = time.time() - start_time2
                    printLog("Mapa de calor generado en: " + str(datetime.timedelta(seconds=int(elapsed_time2))) + "\n")
                    plt.show(block=False)


                DATETIME2 = time.strftime("%Y-%m-%d %H:%M:%S")
                printLog("Programa finalizado en " + DATETIME2 + "\n")
                elapsed_time = time.time() - start_time
                printLog("Tiempo Total Transcurrido: " + str(datetime.timedelta(seconds=int(elapsed_time))) + "\n\n")

                plt.show()
                
                input("Presione Enter para salir...")


                 
