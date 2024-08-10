from utilities.functions import *
image_path = "flag.png"
img = cv2.imread(image_path)
p = 0.6 #reduce the size of the image to reduce computation time
W = int(img.shape[1])
H = int(img.shape[0])
img = cv2.resize(img, (int(img.shape[1] * p), int(img.shape[0] * p)))
w = int(img.shape[1])
h = int(img.shape[0])

# threshold of Canny edge detection
thresh1 = 100
thresh2 = 200
edge = cv2.Canny(img, thresh1, thresh2)
x, y, pts = create_point_list(edge)

adj_matrix = create_adj_matrix(pts, 1, 2)
dict_items = iter(adj_matrix.items())
first_item = next(dict_items)
node_stack = dfs(adj_matrix, first_item[0])
real, imaginary = [node[0] for node in node_stack], [node[1] for node in node_stack]

complex_array = np.vectorize(complex)(real, imaginary)
fourier_array = np.fft.fft(complex_array)

# sample frequency for Fourier Transform
sample_freq = 500
# number of frequencies to be used in reconstruction. this is fun to play with
num_freq = 100
# sizing ratio to counter truncation error while converting to integer
ratio = 10

N = len(complex_array)
T = 1/sample_freq
t = np.linspace(0.0, N*T, N)
tf = np.linspace(0.0, 1.0/T, N)
normalized_array = 1/N * fourier_array
reordered_array, index = reorder_fft_array(normalized_array)

theta = np.angle(reordered_array)
omega = 2*np.pi*index
time_array = np.linspace(0,5,1000)
point_array = [(0, 0)] * len(time_array)
count = 0

for time in time_array:
    canvas = np.zeros((h*ratio, w*ratio, 3), dtype=np.uint8)
    # canvas = cv2.resize(edge.copy(), (int(w*ratio), int(h*ratio)))
    X = 0
    Y = 0
    for i in range(len(index)): #change this to change the number of frequencies used for reconstruction
        radius = abs(reordered_array[i])
        prev_X = X
        prev_Y = Y
        X = X + radius * np.cos(theta[i] + omega[i] * time)
        Y = Y + radius * np.sin(theta[i] + omega[i] * time)
        canvas = cv2.line(canvas, (int(prev_X*ratio), int(prev_Y*ratio)), (int(X*ratio), int(Y*ratio)), (225, 225, 225), int(1*ratio))
    point_array[count] = (int(X*ratio), int(Y*ratio))
    for i in range(len(time_array)):
        canvas = cv2.circle(canvas, point_array[i], int(2*ratio), (225, 225, 225), -1)
    count += 1
    canvas = cv2.resize(canvas, (w, h))
    cv2.imshow("", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
