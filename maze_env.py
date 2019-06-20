
import numpy as np
import tensorflow as tf
import time
import subprocess
import sys
import io
import random
import math
import os
from skimage import io,transform


from PIL import Image
import random
from six.moves import input

from common.auto_adb import auto_adb
from common import debug, config, UnicodeStreamFilter

adb = auto_adb()
config = config.open_accordant_config()
debug_switch = True  # debug 开关，需要调试的时候请改为：True

VERSION = "1.1.4"

# Magic Number，不设置可能无法正常执行，请根据具体截图从上到下按需设置，设置保存在 config 文件夹中
under_game_score_y = config['under_game_score_y']
# press_coefficient = config['press_coefficient']  # 长按的时间系数，请自己根据实际情况调节
piece_base_height_1_2 = config['piece_base_height_1_2']  # 二分之一的棋子底座高度，可能要调节
piece_body_width = config['piece_body_width']  # 棋子的宽度，比截图中量到的稍微大一点比较安全，可能要调节

screenshot_way = 2

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

class env(object):
    def __init__(self):
        self.lastScore = 0

    def pull_screenshot(self):
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        if sys.platform == 'win32':
            screenshot = screenshot.replace(b'\r\n', b'\n')
        f = open('autojump.png', 'wb')
        f.write(screenshot)
        f.close()

    def pull_screenshot_temp(self):
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        if sys.platform == 'win32':
            screenshot = screenshot.replace(b'\r\n', b'\n')
        f = open('autojump_temp.png', 'wb')
        f.write(screenshot)
        f.close()

    def pull_screenshot_afterJump(self):
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        screenshot = process.stdout.read()
        if sys.platform == 'win32':
            screenshot = screenshot.replace(b'\r\n', b'\n')
        f = open('after_autojump_temp.png', 'wb')
        f.write(screenshot)
        f.close()

    def set_button_position(self, im):
        """
        将 swipe 设置为 `再来一局` 按钮的位置
        """
        global swipe_x1, swipe_y1, swipe_x2, swipe_y2
        w, h = im.size
        left = int(w / 2)
        top = int(1584 * (h / 1920.0))
        left = int(random.uniform(left - 100, left + 100))
        top = int(random.uniform(top - 100, top + 100))  # 随机防 ban
        after_top = int(random.uniform(top - 100, top + 100))
        after_left = int(random.uniform(left - 100, left + 100))
        swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, after_left, after_top

    def jump(self, distance, press_coefficient):
        '''
        跳跃一定的距离
        '''
        # if ai.get_result_len() >= 10:  # 需采集10条样本以上
        #     k, b, v = ai.computing_k_b_v(distance)
        #     press_time = distance * k[0] + b
        #     print('Y = {k} * X + {b}'.format(k=k[0], b=b))
        #
        # else:
        press_time = distance * press_coefficient
        press_time = max(press_time, 200)  # 设置 200ms 是最小的按压时间

        press_time = int(press_time)
        cmd = 'shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
            x1=swipe_x1,
            y1=swipe_y1,
            x2=swipe_x2,
            y2=swipe_y2,
            duration=press_time
        )
        print('{}'.format(cmd))
        adb.run(cmd)
        return press_time

    def find_piece(self, im):
        '''
        寻找关键坐标
        '''
        w, h = im.size

        piece_x_sum = 0
        piece_x_c = 0
        piece_y_max = 0
        scan_x_border = int(w / 8)  # 扫描棋子时的左右边界
        scan_start_y = 0  # 扫描的起始 y 坐标
        im_pixel = im.load()
        # 以 50px 步长，尝试探测 scan_start_y
        for i in range(int(h / 3), int(h * 2 / 3), 50):
            last_pixel = im_pixel[0, i]
            for j in range(1, w):
                pixel = im_pixel[j, i]
                # 不是纯色的线，则记录 scan_start_y 的值，准备跳出循环
                if pixel[0] != last_pixel[0] or pixel[1] != last_pixel[1] or pixel[2] != last_pixel[2]:
                    scan_start_y = i - 50
                    break
            if scan_start_y:
                break
        # print('scan_start_y: {}'.format(scan_start_y))

        # 从 scan_start_y 开始往下扫描，棋子应位于屏幕上半部分，这里暂定不超过 2/3
        for i in range(scan_start_y, int(h * 2 / 3)):
            for j in range(scan_x_border, w - scan_x_border):  # 横坐标方面也减少了一部分扫描开销
                pixel = im_pixel[j, i]
                # 根据棋子的最低行的颜色判断，找最后一行那些点的平均值，这个颜色这样应该 OK，暂时不提出来
                if (50 < pixel[0] < 60) and (53 < pixel[1] < 63) and (95 < pixel[2] < 110):
                    piece_x_sum += j
                    piece_x_c += 1
                    piece_y_max = max(i, piece_y_max)

        if not all((piece_x_sum, piece_x_c)):
            return 0, 0,
        piece_x = int(piece_x_sum / piece_x_c)
        piece_y = piece_y_max - piece_base_height_1_2  # 上移棋子底盘高度的一半

        return piece_x, piece_y

    def find_end_piece_and_current_board(self, im):
        # 获取image的长宽
        w, h = im.size

        piece_x_sum = 0
        piece_x_c = 0
        piece_y_max = 0
        board_x = 0
        board_y = 0

        left_value = 0
        left_count = 0
        right_value = 0
        right_count = 0
        from_left_find_board_y = 0
        from_right_find_board_y = 0

        scan_x_border = int(w / 8)  # 扫描棋子时的左右边界
        scan_start_y = 0  # 扫描的起始y坐标
        # 32位彩色图像，24bit表示红色、绿色和蓝色三个通道
        im_pixel = im.load()

        # 以50px步长，尝试探测scan_start_y
        for i in range(int(h / 3), int(h * 2 / 3), 50):
            last_pixel = im_pixel[0, i]
            for j in range(1, w):
                pixel = im_pixel[j, i]
                # 不是纯色的线，则记录scan_start_y的值，准备跳出循环
                if pixel[0] != last_pixel[0] or pixel[1] != last_pixel[1] or pixel[2] != last_pixel[2]:
                    scan_start_y = i - 50
                    break
            if scan_start_y:
                break
        print('scan_start_y: ', scan_start_y)

        # 从scan_start_y开始往下扫描，棋子应位于屏幕上半部分，这里暂定不超过2/3
        for i in range(scan_start_y, int(h * 2 / 3)):
            for j in range(scan_x_border, w - scan_x_border):  # 横坐标方面也减少了一部分扫描开销
                pixel = im_pixel[j, i]
                # 根据棋子的最低行的颜色判断，找最后一行那些点的平均值，这个颜色这样应该 OK，暂时不提出来
                if (50 < pixel[0] < 60) and (53 < pixel[1] < 63) and (95 < pixel[2] < 110):
                    piece_x_sum += j
                    piece_x_c += 1
                    piece_y_max = max(i, piece_y_max)

        if not all((piece_x_sum, piece_x_c)):
            return 0, 0, 0, 0
        piece_x = piece_x_sum / piece_x_c
        piece_y = piece_y_max - piece_base_height_1_2  # 上移棋子底盘高度的一半


    def find_pece_ad_board2(self, im):
        # ===== 先找 ===
        pass


    def find_piece_and_board(self, im):
        # 获取image的长宽
        w, h = im.size

        piece_x_sum = 0
        piece_x_c = 0
        piece_y_max = 0
        board_x = 0
        board_y = 0

        left_value = 0
        left_count = 0
        right_value = 0
        right_count = 0
        from_left_find_board_y = 0
        from_right_find_board_y = 0

        scan_x_border = int(w / 8)  # 扫描棋子时的左右边界
        scan_start_y = 0  # 扫描的起始y坐标
        # 32位彩色图像，24bit表示红色、绿色和蓝色三个通道
        im_pixel = im.load()

        # 以50px步长，尝试探测scan_start_y

        for i in range(int(h / 3), int(h * 2 / 3), 50):
            last_pixel = im_pixel[0, i]
            for j in range(1, w):
                pixel = im_pixel[j, i]
                # 不是纯色的线，则记录scan_start_y的值，准备跳出循环
                if pixel[0] != last_pixel[0] or pixel[1] != last_pixel[1] or pixel[2] != last_pixel[2]:
                    scan_start_y = i - 50
                    break
            if scan_start_y:
                break
        print('scan_start_y: ', scan_start_y)


        # 从scan_start_y开始往下扫描，棋子应位于屏幕上半部分，这里暂定不超过2/3
        for i in range(scan_start_y, int(h * 2 / 3)):
            for j in range(scan_x_border, w - scan_x_border):  # 横坐标方面也减少了一部分扫描开销
                pixel = im_pixel[j, i]
                # 根据棋子的最低行的颜色判断，找最后一行那些点的平均值，这个颜色这样应该 OK，暂时不提出来
                if (50 < pixel[0] < 60) and (53 < pixel[1] < 63) and (95 < pixel[2] < 110):
                    piece_x_sum += j
                    piece_x_c += 1
                    piece_y_max = max(i, piece_y_max)

        if not all((piece_x_sum, piece_x_c)):
            return 0, 0, 0, 0
        piece_x = piece_x_sum / piece_x_c
        piece_y = piece_y_max - piece_base_height_1_2  # 上移棋子底盘高度的一半

        for i in range(int(h / 3), int(h * 2 / 3)):


            last_pixel = im_pixel[0, i]
            # 计算阴影的RGB值,通过photoshop观察,阴影部分其实就是背景色的明度V 乘以0.7的样子
            h, s, v = rgb2hsv(last_pixel[0], last_pixel[1], last_pixel[2])
            r, g, b = hsv2rgb(h, s, v * 0.7)

            if from_left_find_board_y and from_right_find_board_y:
                break

            if not board_x:
                board_x_sum = 0
                board_x_c = 0

                for j in range(w):
                    pixel = im_pixel[j, i]
                    # 修掉脑袋比下一个小格子还高的情况的 bug
                    if abs(j - piece_x) < piece_body_width:
                        continue

                    # 修掉圆顶的时候一条线导致的小 bug，这个颜色判断应该 OK，暂时不提出来
                    if abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(
                            pixel[2] - last_pixel[2]) > 10:
                        board_x_sum += j
                        board_x_c += 1
                if board_x_sum:
                    board_x = board_x_sum / board_x_c
            else:
                # 继续往下查找,从左到右扫描,找到第一个与背景颜色不同的像素点,记录位置
                # 当有连续3个相同的记录时,表示发现了一条直线
                # 这条直线即为目标board的左边缘
                # 然后当前的 y 值减 3 获得左边缘的第一个像素
                # 就是顶部的左边顶点
                for j in range(w):
                    pixel = im_pixel[j, i]
                    # 修掉脑袋比下一个小格子还高的情况的 bug
                    if abs(j - piece_x) < piece_body_width:
                        continue
                    if (abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2])
                        > 10) and (abs(pixel[0] - r) + abs(pixel[1] - g) + abs(pixel[2] - b) > 10):
                        if left_value == j:
                            left_count = left_count + 1
                        else:
                            left_value = j
                            left_count = 1

                        if left_count > 3:
                            from_left_find_board_y = i - 3
                        break
                # 逻辑跟上面类似,但是方向从右向左
                # 当有遮挡时,只会有一边有遮挡
                # 算出来两个必然有一个是对的
                for j in range(w)[::-1]:
                    pixel = im_pixel[j, i]
                    # 修掉脑袋比下一个小格子还高的情况的 bug
                    if abs(j - piece_x) < piece_body_width:
                        continue
                    if (abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2])
                        > 10) and (abs(pixel[0] - r) + abs(pixel[1] - g) + abs(pixel[2] - b) > 10):
                        if right_value == j:
                            right_count = left_count + 1
                        else:
                            right_value = j
                            right_count = 1

                        if right_count > 3:
                            from_right_find_board_y = i - 3
                        break

        # 如果顶部像素比较多,说明图案近圆形,相应的求出来的值需要增大,这里暂定增大顶部宽的三分之一
        if board_x_c > 5:
            from_left_find_board_y = from_left_find_board_y + board_x_c / 3
            from_right_find_board_y = from_right_find_board_y + board_x_c / 3

        # 按实际的角度来算，找到接近下一个 board 中心的坐标 这里的角度应该是30°,值应该是tan 30°,math.sqrt(3) / 3
        board_y = piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3

        # 从左从右取出两个数据进行对比,选出来更接近原来老算法的那个值
        if abs(board_y - from_left_find_board_y) > abs(from_right_find_board_y):
            new_board_y = from_right_find_board_y
        else:
            new_board_y = from_left_find_board_y

        if not all((board_x, board_y)):
            return 0, 0, 0, 0

        return piece_x, piece_y, board_x, new_board_y

    def check_screenshot(self):
        '''
        检查获取截图的方式
        '''
        global screenshot_way
        if os.path.isfile('autojump.png'):
            os.remove('autojump.png')
        if (screenshot_way < 0):
            print('暂不支持当前设备')
            sys.exit()
        self.pull_screenshot()
        try:
            Image.open('./autojump.png').load()
            print('采用方式 {} 获取截图'.format(screenshot_way))
        except Exception:
            screenshot_way -= 1
            self.check_screenshot()

    def yes_or_no(self, prompt, true_value='y', false_value='n', default=True):
        default_value = true_value if default else false_value
        prompt = '%s %s/%s [%s]: ' % (prompt, true_value, false_value, default_value)
        i = input(prompt)
        if not i:
            return default
        while True:
            if i == true_value:
                return True
            elif i == false_value:
                return False
            prompt = 'Please input %s or %s: ' % (true_value, false_value)
            i = input(prompt)

    def getReward(self, target_distance, actual_distance, ActualScore):

        error_distance = math.sqrt((target_distance - actual_distance) ** 2)
        # 利用踩中中心作为奖励，因为移动后的估算会有问题，所以不能用估算后的位置来评估
        if ActualScore - self.lastScore > 1:
            return 300
        else:
            return -1 * (error_distance)

        # RawReward = 100.0 / (abs(target_distance - actual_distance) + 1)
        # # 力度不够，需要调高力度
        # if target_distance > actual_distance:
        #     if action == 2:
        #         Reward = 1
        #     else:
        #         Reward = -1
        # elif target_distance == actual_distance:
        #     if action == 1:
        #         Reward = 10
        #     else:
        #         Reward = -1
        # else:
        #     if action == 0:
        #         Reward = 1
        #     else:
        #         Reward = -1
        # return Reward * RawReward

    def strint(self, score0):
        if (score0 < 10):
            return str(score0)
        else:
            return ""

    def pixel_division(self, img, w, h):
        pixels = list(img.getdata())
        row_pix = np.zeros([1, h])
        col_pix = np.zeros([1, w])
        for i in range(w):
            for j in range(h):
                if pixels[j * w + i] < 100:
                    row_pix[0, j] += 1
                    col_pix[0, i] += 1
        start_h = 0
        end_h = 0
        flag = 0
        for j in range(h):
            if row_pix[0, j] >= 1 and flag == 0:
                start_h = j
                flag = 1
            if row_pix[0, j] >= 1:
                end_h = j

        pixels_Widh = []
        end_w = 0
        for i in range(1, w):
            if col_pix[0, i - 1] <= 0 and col_pix[0, i] >= 1:
                pixels_Widh.append(i - 1)
            if col_pix[0, i] >= 1:
                end_w = i
        pixels_Widh.append(end_w + 1)
        return start_h, end_h, pixels_Widh

    def pross_data(self, image):
        pixels = list(image.getdata())  # 得到像素数据 灰度0-255
        # print(len(pixels))
        for i in range(len(pixels)):
            if pixels[i] < 100:
                pixels[i] = 0
            else:
                pixels[i] = 255
        return pixels

    def read_one_image(self, path):
        img = io.imread(path)
        w = 81
        h = 81
        c = 1
        img = transform.resize(img, (w, h, c))
        return np.asarray(img)

    def getScore(self, im):
        with tf.Session() as sess:

            saver = tf.train.import_meta_graph('./resource/model/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./resource/model/'))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            logits = graph.get_tensor_by_name("logits_eval:0")

            ##比例系数
            pix_w = im.size[0] * 1.0 / 1080
            pix_h = im.size[1]
            region = im.crop((0, pix_h * 0.1, 460 * pix_w, pix_h * 0.2))
            region = region.convert('L')
            start_h, end_h, pixels_Widh = self.pixel_division(region, int(460 * pix_w), int(pix_h * 0.1))
            data = []
            for i in range(len(pixels_Widh) - 1):
                region1 = region.crop((pixels_Widh[i], start_h, pixels_Widh[i + 1], end_h))
                region1.putdata(self.pross_data(region1))
                str1 = "./region" + str(i) + ".png"
                region1.save(str1)
                data1 = self.read_one_image(str1)
                data.append(data1)
            feed_dict = {x: data}
            classification_result = sess.run(logits, feed_dict)
            output = []
            output = tf.argmax(classification_result, 1).eval()
            m_score = ""
            for i in range(len(output)):
                m_score += self.strint(output[i])
            m_score = int(m_score)
            print('score:{}'.format(m_score))
        return m_score

    def observeDistance(self):
        # 截屏，发现当前的状态，截屏为autojump.png
        self.pull_screenshot()
        im = Image.open('./autojump.png')

        # 获取棋子和 board 的位置
        piece_x, piece_y, board_x, board_y = self.find_piece_and_board(im)
        ts = int(time.time())
        # print(ts, piece_x, piece_y, board_x, board_y)
        self.set_button_position(im)
        distance = math.sqrt((board_x - piece_x) ** 2 + (board_y - piece_y) ** 2)
        return im, distance, piece_x, piece_y, board_x, board_y, ts

    # action 只有三种：分别是增加，不变和减少。
    # 触发action，得到observation_, reward
    # action 到 press_coefficient这个变换，是需要
    def step(self, press_coefficient, action):

        im, distance, piece_x, piece_y, board_x, board_y, ts = self.observeDistance()
        press_time = self.jump(distance, press_coefficient)

        # 在跳跃落下的瞬间 摄像机移动前截图 这个参数要自己校调
        time.sleep(0.07)
        self.pull_screenshot_temp()
        im_temp = Image.open('./autojump_temp.png')
        temp_piece_x, temp_piece_y = self.find_piece(im_temp)
        # temp_piece_x, temp_piece_y, temp_board_x, temp_board_y = self.find_piece_and_board(im_temp)


        # # 获取它的分数
        time.sleep(1)
        self.pull_screenshot_afterJump()
        after_im_temp = Image.open('./after_autojump_temp.png')
        # after_piece_x, after_piece_y = self.find_piece(after_im_temp)

        target_distance, actual_distance = debug.computing_error(press_time, board_x, board_y, piece_x, piece_y, temp_piece_x, temp_piece_y)


        if debug_switch:
            debug.save_debug_screenshot2(ts, im, piece_x, piece_y, board_x, board_y, temp_piece_x, temp_piece_y)
            debug.save_debug_screenshot('raw:' + str(ts), im_temp, temp_piece_x, temp_piece_y, board_x, board_y)
            # debug.save_debug_screenshot('after_jump: ' + str(ts), after_im_temp, piece_x, piece_y, board_x, board_y)

            # after_im_temp = Image.open('./after_autojump_temp.png')

            # debug.save_debug_screenshot('raw:' + str(ts), im, piece_x, piece_y, board_x, board_y)
            # debug.save_debug_screenshot(ts, im_temp, board_x, board_y, temp_piece_x, temp_piece_y)
            # debug.backup_screenshot(ts)

        time.sleep(random.uniform(0.5, 0.6))  # 为了保证截图的时候应落稳了，多延迟一会儿，随机值防 ban

        # 真实的得分记录
        ActualScore = self.getScore(after_im_temp)

        # observation_ = np.array([1.3])
        reward = self.getReward(target_distance, actual_distance, ActualScore)
        self.lastScore = ActualScore
        return reward










