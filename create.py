import landscape_model


def make_image(img_tensor):
  img_tensor = img_tensor.cpu().detach().numpy()
  ret = list()
  for i in range(len(img_tensor[0])):
    ret.append(list())
    for j in range(len(img_tensor[0][0])):
      ret[-1].append([img_tensor[0][i][j],img_tensor[1][i][j],img_tensor[2][i][j]])
  return ret



test_color = resize(np.array(Image.open(r'test_color.JPG')),(256,256,3))
test_outline = resize(np.array(Image.open(r'test_outline.JPG')),(256,256,3))

test_outline = test_outline[:,:,0]*0.3+test_outline[:,:,1]*0.587+test_outline[:,:,2]*0.114
test_outline = (test_outline < 0.9).astype(int)

test_color_blur = ndi.uniform_filter(test_color, size=(50, 50, 1))
test_color_blur = ndi.uniform_filter(test_color_blur, size=(10, 10, 1))

plt.imshow(test_color)
plt.show()
#plt.imshow(feature.canny(test_outline[:,:,0] + test_outline[:,:, sigma=2))
plt.imshow(test_outline)
plt.show()
plt.imshow(test_color)
test_color_a = test_color_blur

nn_test_input = torch.from_numpy(np.array([[test_outline,test_color_a[:,:,0],test_color_a[:,:,1], test_color_a[:,:,2]]])).float() * 2 - 1
fake_test = landscape_model.Generator(nn_test_input)

plt.imshow(np.array(make_image(fake_test[0])) * 0.5 + 0.5)
plt.show()