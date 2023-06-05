% test.m
% use this file to calulate ssim
% it also calculates psnr
N = 512;
output_img_path = '/Users/np/Desktop/test/*.flt';
%ref_img_path = '/Users/np/Desktop/unet/UNet_results/testoutput/sparseview_60/*.flt';
ref_img_path =  '/Users/np/Desktop/CT_data/images/ndct/test/*.flt';
files_ref = dir(ref_img_path);

files_output = dir(output_img_path);

%assumes files will be listed in same order

M = numel(files_ref);
Z = numel(files_output);

psnr_vals = zeros(1,M);
ssim_vals = zeros(1,M);
for m = 1:M
    fid = fopen([files_ref(m).folder '/' files_ref(m).name],'r');
    img_ref = fread(fid,N*N,'float');
    fclose(fid);
    fid = fopen([files_output(m).folder '/' files_output(m).name],'r');
    img_output = fread(fid,N*N,'float');
    fclose(fid);
    
    maxval = max(img_ref(:));
    
    p = psnr(img_output,img_ref,maxval);
    s = ssim(img_output,img_ref,'DynamicRange',maxval);
    
    fprintf('Img #%d: PSNR = %2.2f\tSSIM = %1.4f\n',m,p,s);
    psnr_vals(m) = p; ssim_vals(m) = s;
end
fprintf('\n Avg PSNR: %2.2f\tAvg SSIM: %1.4f\n',mean(psnr_vals),mean(ssim_vals));
