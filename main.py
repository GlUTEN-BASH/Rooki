import pickle
from random import randint
import cv2
import mediapipe as mp
import numpy as np
import pygame

pygame.init()
pygame.mixer.init()

theme = 0

HEIGHT = 650
WIDTH = 170
screen = pygame.display.set_mode((HEIGHT, WIDTH))
clock = pygame.time.Clock()
pygame.display.set_caption('Нейрожест Pre-Release')
icon = pygame.image.load('icon.png')
pygame.display.set_icon(icon)
FPS = 60
running = False
settings = True

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

camnum = 0 


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

game_mode = 0
key = 0
flag_pass = 0
word_pass = 0
letter = "А"
letterbin = []
score = 0
chtick = 0
counter = 0
scene_counter = 0
letcount = 0
word = 0
letf = 0
progress = []
character = 'А'

fl = False
oldmode = False
zerk = False

astra = (197, 121, 255)
astratext = (197, 255, 100)
white = (255, 255, 255)
black = (0, 0, 0)

my_font = pygame.font.SysFont('verdana', 30) 
bigaboom = pygame.font.SysFont('verdana', 70) 

text = black
back = white


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


words = ['НЕТ', 'СТУЛ', 'АГА', 'ЖЕЛЕЗО', 'Я', 'ТЫ', 'ДОМ', 'ЭРА', 'ПОТОМ', 'СИЛА', 'САДИЗМ', 'РЫБА', 'РАБОТА', 'ЮНГА', 'ДУБ', 'ПАПА', 'МАМА', 'МОНО', 'СТЕРЕО', 'ВИНИЛ', 'СТВОЛ', 'ТВ', 'ФУ', 'ЦЕНТ', 'ФУНТ', 'МИЛО', 'ФМ', 'РАДИО', 'СОДА', 'ЗУБ', 'ГАММА', 'ГЫЧА', 'БОБС']

labels_dict = {0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ж', 7: 'З', 8: 'И', 9: 'Л', 10: 'М', 11: 'Н', 12: 'О', 13: 'П', 14: 'Р', 15: 'С', 16: 'Т', 17: 'У', 18: 'Ф',  19: 'Ц',  20: 'Ч',  21: 'Ш', 22:'Ь', 23:'Ы', 24: 'Э', 25: 'Ю', 26: 'Я'}

predbin = {}

for i in labels_dict:
    predbin.update({labels_dict[i]: 0})

while settings: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            settings = False
            running = True
        if event.type == pygame.KEYDOWN:
            match event.key:
                case pygame.K_0:
                    camnum = 0
                case pygame.K_1:
                    camnum = 1
                case pygame.K_2:
                    camnum = 2
                case pygame.K_3:
                    camnum = 3
                case pygame.K_4:
                    camnum = 4
                case pygame.K_5:
                    camnum = 5
                case pygame.K_a:
                    theme = 1
                case pygame.K_f:
                    fl = True
                case pygame.K_b:
                    oldmode = True
                case pygame.K_v:
                    zerk = True
                case pygame.K_ESCAPE:
                    settings = False
                    running = True

    screen.fill(back)
    screen.blit(my_font.render('0 - 5 - Выбор камеры', False, (text)), (0, 0)) 
    screen.blit(my_font.render('Нажмите ESC для продолжения', False, (text)), (0, 30))
    screen.blit(my_font.render('Нажмите F для отзеркаливания камеры', False, (text)), (0, 60))
    screen.blit(my_font.render('Нажмите B для вывода скелета', False, (text)), (0, 90))
    screen.blit(my_font.render('Нажмите V для отзеркаливания вывода', False, (text)), (0, 120))
    pygame.display.flip()

cap = cv2.VideoCapture(camnum)

if theme == 0:
    text = black
    back = white
elif theme == 1:
    text = astratext
    back = astra

HEIGHT = 1200
WIDTH = 450

screen = pygame.display.set_mode((HEIGHT, WIDTH))

while running: 
    try:
        clock.tick(FPS)
        data_aux = []
        x_ = []
        y_ = [] 

        ret, frame = cap.read()
        if fl:
            frame = cv2.flip(frame, 1)         
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

        if chtick != 7:
            chtick += 1
            predbin[predicted_character] += 1
        else:
            chtick = 0

            v = list(predbin.values())
            k = list(predbin.keys())
            character = k[v.index(max(v))]

            chtick = 0

            predbin = {}

            for i in labels_dict:
                predbin.update({labels_dict[i]: 0})
                    

        screen.fill((back)) 

        if oldmode:
            frame_py = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_py = np.rot90(frame_py)
        else:
            frame_py = np.rot90(frame_rgb)


        fra = pygame.surfarray.make_surface(frame_py)
        fra = pygame.transform.scale(fra, (640, 480)) 
        if not zerk:
            fra = pygame.transform.flip(fra, True, False) 
        screen.blit(fra, (0,0))
        pygame.draw.rect(screen, back, pygame.Rect(10, 10, 70, 75))
        screen.blit(bigaboom.render(character, False, (text)), (12,0))
        
                         
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        game_mode = 0
                    case pygame.K_1:
                        game_mode = 1
                    case pygame.K_2:
                        game_mode = 2
                    case pygame.K_3:
                        game_mode = 3
                        tutorial = cv2.VideoCapture("./data/Tutorial.mp4")
                        fps = tutorial.get(cv2.CAP_PROP_FPS)
                    case pygame.K_4:
                        game_mode = 4
                    case pygame.K_5:
                        running = False

        if game_mode == 0:
            screen.blit(my_font.render('Добро пожаловать!', False, (text)), (650,0))
            screen.blit(my_font.render('1 - Тренировка', False, (text)), (650,30))
            screen.blit(my_font.render('2 - Печать - тренировка', False, (text)), (650,60))
            screen.blit(my_font.render('3 - Обучение', False, (text)), (650,90))
            screen.blit(my_font.render('4 - Тренировка с подсказками', False, (text)), (650,120))
            screen.blit(my_font.render('5 - Выход', False, (text)), (650,150))
            key = 0
            flag_pass = 0
            letter = "А"
            letterbin = []
            score = 0
            counter = 0
            scene_counter = 0
            letcount = 0             
            word = 0     
            letf = 0

        if game_mode == 1:
            if flag_pass == 0:
                screen.blit(my_font.render(f'Покажите букву: {letter}', False, (text)), (650,0))     
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (650,30))   

                letterbin.append(character)

                for i in letterbin:
                    letcount += 1
                    if i == letter:
                        counter += 1

                if letcount >= 4000:
                    flag_pass = 2

                if counter >= 300:
                    flag_pass = 1
                    counter = 0

            if flag_pass == 1:
                pygame.mixer.music.load(".\data\yes.wav") 
                pygame.mixer.music.play()
                letter = labels_dict[randint(0, 26)]
                flag_pass = 0
                score += 1
                counter = 0
                letterbin = []
                letcount = 0
                
            if flag_pass == 2:
                for i in letterbin:
                    if i == letter:
                        counter += 1

                if scene_counter == 0:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")
                    pygame.mixer.music.load("./data/no.wav") 
                    pygame.mixer.music.play()
                screen.blit(my_font.render(f'GAME OVER', False, (text)), (900,0))     
                letterbin.append(character)
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (900,30))   
                scene_counter += 1
                try:
                    letfps = vid.get(cv2.CAP_PROP_FPS)
                    success, video_image = vid.read()
                    video_surf = pygame.image.frombuffer(video_image.tobytes(), video_image.shape[1::-1], "BGR")
                    screen.blit(pygame.transform.scale(video_surf, (250, 450)), (640, 0))
                    if counter >= 600:
                        game_mode = 0
                        pygame.mixer.music.load(".\data\yes.wav") 
                        pygame.mixer.music.play()   
                except:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")



        if game_mode == 2:
            letter = words[word][letf]
            if flag_pass == 0:
                screen.blit(my_font.render(f'Покажите букву: {letter}', False, (text)), (650,0))     
                letterbin.append(character)
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (650,30))   
                screen.blit(my_font.render(f'Слово: {words[word]}', False, (text)), (650,60))  
                screen.blit(my_font.render(f'Прогресс: ', False, (text)), (650,90))  
                screen.blit(my_font.render(f''.join(progress), False, (text)), (810, 90))  
                
                for i in letterbin:
                    letcount += 1
                    if i == letter:
                        counter += 1

                if letcount >= 4000:
                    flag_pass = 2

                if counter >= 300:
                    flag_pass = 1
                    counter = 0

            if flag_pass == 1:
                pygame.mixer.music.load(".\data\yes.wav") 
                pygame.mixer.music.play()
                if letf >= len(words[word]) - 1:
                    letf = 0
                    flag_pass = 0
                    counter = 0
                    letterbin = []
                    letcount = 0
                    progress = []
                    word = randint(0, len(words))    
                    score += 1
                    word_pass = 0
                    pass
                else:
                    progress.append(letter)
                    letf += 1
                    flag_pass = 0
                    counter = 0
                    letterbin = []
                    letcount = 0      
                letter = words[word][letf]


                
            if flag_pass == 2:
                for i in letterbin:
                    if i == letter:
                        counter += 1

                if scene_counter == 0:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")
                    pygame.mixer.music.load("./data/no.wav") 
                    pygame.mixer.music.play()
                screen.blit(my_font.render(f'GAME OVER', False, (text)), (900,0))     
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (900,30))   
                scene_counter += 1
                lettersss = list(labels_dict.values())
                
                try:
                    letfps = vid.get(cv2.CAP_PROP_FPS)
                    success, video_image = vid.read()
                    video_surf = pygame.image.frombuffer(video_image.tobytes(), video_image.shape[1::-1], "BGR")
                    screen.blit(pygame.transform.scale(video_surf, (250, 450)), (640, 0))
                    if counter >= 600:
                        game_mode = 0
                        pygame.mixer.music.load(".\data\yes.wav") 
                        pygame.mixer.music.play()   
                except:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")



        if game_mode == 4:
            if flag_pass == 0:
                im = pygame.image.load(f"./data/{letter}.png")
                pygame.transform.scale(im, (200, 350))
                screen.blit(im, (650, 0))
                screen.blit(my_font.render(f'Покажите букву: {letter}', False, (text)), (860,0))     
                letterbin.append(character)
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (860,30))   
                for i in letterbin:
                    letcount += 1
                    if i == letter:
                        counter += 1

                if letcount >= 4000:
                    flag_pass = 2

                if counter >= 300:
                    flag_pass = 1
                    counter = 0

            if flag_pass == 1:
                pygame.mixer.music.load(".\data\yes.wav") 
                pygame.mixer.music.play()
                letter = labels_dict[randint(0, 26)]
                flag_pass = 0
                score += 1
                counter = 0
                letterbin = []
                letcount = 0
                
            if flag_pass == 2:
                for i in letterbin:
                    if i == letter:
                        counter += 1

                if scene_counter == 0:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")
                    pygame.mixer.music.load("./data/no.wav") 
                    pygame.mixer.music.play()
                screen.blit(my_font.render(f'GAME OVER', False, (text)), (900,0))     
                letterbin.append(character)
                screen.blit(my_font.render(f'Очки: {str(score)}', False, (text)), (900,30))   
                scene_counter += 1
                lettersss = list(labels_dict.values())
                try:
                    letfps = vid.get(cv2.CAP_PROP_FPS)
                    success, video_image = vid.read()
                    video_surf = pygame.image.frombuffer(video_image.tobytes(), video_image.shape[1::-1], "BGR")
                    screen.blit(pygame.transform.scale(video_surf, (250, 450)), (640, 0))
                    if counter >= 600:
                        game_mode = 0
                        pygame.mixer.music.load(".\data\yes.wav") 
                        pygame.mixer.music.play()   
                except:
                    vid = cv2.VideoCapture(f"./data/{letter}.mp4")

        if game_mode == 3:
            success, video_image = tutorial.read()
            video_surf = pygame.image.frombuffer(video_image.tobytes(), video_image.shape[1::-1], "BGR")
            screen.blit(pygame.transform.scale(video_surf, (250, 450)), (640, 0))

        pygame.display.flip()             
    except Exception as e:
        pass

pygame.quit()


cap.release()
cv2.destroyAllWindows()