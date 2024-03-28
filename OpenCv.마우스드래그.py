import cv2 as cv
import sys

img = cv.imread('apples.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event, x, y, flags, param): #콜백 함수
    global ix, iy

    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭했을 떄 초기 위치 저장
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 클릭했을 떄 직사각형 그리기
        cv.rectangle(img, (ix, iy), (x, y), (0,0,255), 2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw) # Drawing 윈도우에 draw 콜백 함수 지정

while(True):       # 마우스 이벤트가 언제 발생할지 모르므로 무한 반복
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break