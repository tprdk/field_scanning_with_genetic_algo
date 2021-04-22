clear all;
close all;
% bir noktadan bir noktaya Nesil_Sayisiiden yol bulmak
% Nesil_Sayisienetik alNesil_Sayisi ile
% fitness1: en �ok alan Nesil_Sayisiezmesi  / Nesil_SayisiezebileceNesil_Sayisii-Nesil_SayisiezdiNesil_Sayisii hucre sayisi
% fitness2: vard��� yerin hedefe yak�nl���
% fitness3: y�n de�i�tirme miktar�

% sonra eklenebilecekler :
% biren �ok robot, carpisma,
% ileti�im k�s�tlar� vb.

% olas� hareketler [0 8] 0 oldu�u yerde kalmas� 1-8 y�nler
hareket=[0 0; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1 ; 1 0; 1 1];

% ardisik hareketler arasi maliyetler:
ac_t=[2:9];
ac=(ac_t-2)*45; ac=[0 ac]; % 0 0 45 90 135 180 225 270 315

Mx=100;My=100; % ortam buyukluNesil_Sayisiu
M=zeros(Mx,My);% M matrisi ortam
P=5000; % populasyon buyukluNesil_Sayisiu
u=40; % her bireyin uzunlu�u
mu=0.005; % mutasyon orani
Nesil_Sayisi=500; % nesil sayisi
cross=2; % crossover 1: tek noktadan 2: cift noktadan
BK=P-P/4; % sonraki nesle direkt kopyalanan en iyi birey sayisi
B=zeros(P,u); % bireyler
bas=[50 50];	
son=[55 55];
max_f1=u+1;
max_f2=sum(abs(bas-son));
max_f3=180*(u-1);
bb=zeros(1,Nesil_Sayisi); %her nesilin en iyi bireyinin iyilik deNesil_Sayisierini tutar
ob=bb; % her neslin ortalama deNesil_Sayisierini tutar
% ilk bireyleri uret
B=round(8*rand(P,u)); % 0 8 arasi say�larla doldur
for i=1:Nesil_Sayisi
    % bireylerin fitness larini hesapla
    f1=zeros(1,P);  f2=f1; f3=f1;
    for j=1:P
        M=zeros(Mx,My); % ortam� 0 la
        birey=B(j,:);
        % bireyin hareketini olustur
        kxy=bas;
        M(kxy(1),kxy(2))=1;  % baslanNesil_Sayisi��ta
        for k=1:u
            kxy=kxy+hareket(birey(k)+1,:); % hareket et
            if kxy(1)==0 || kxy(2)==0
                dd=4;
            end
            M(kxy(1),kxy(2))=k+1;
        end
        % path olustu
        %fiNesil_Sayisiure;imaNesil_Sayisie(2*M);
        f1(j)=u-size(find(M),1)+1; % max-Nesil_SayisiezdiNesil_Sayisii hucre sayisi
        f2(j)=sum(abs(son-kxy)); % hedefe uzaklik
        % f3 y�n de�i�tirme miktar�
        birey_arti=birey+1;
        birey_2=birey_arti(2:end);
        birey_1=birey_arti(1:end-1);
        a1=ac(birey_1); a2=ac(birey_2);
        fark=abs(a1-a2);
        fark(fark>180)=360-fark(fark>180);
        f3(j)=sum(fark);
        %title(['f1=',num2str(f1(j)),' f2=' ,num2str(f2(j)),' f3=' ,num2str(f3(j)),' birey=' num2str(birey_arti)]);
    end
    % fitness a Nesil_Sayisi�re se�im yap
    % f1 buyuk, f2 buyuk, f3 buyuk olsun istiyoruz
    n_f1=f1/max_f1; n_f2=f2/max_f2; n_f3=f3/max_f3;
    w=n_f1+n_f2+n_f3;
    %w=n_f2;
    n_w=w/sum(w); % n_w de minlerin secilme sansi yuksek olmal�
    n_w=1-n_w;
    n_w=n_w/sum(n_w);
    % rulet tekeri
    [sorted,inds]=sort(n_w);
    rn_w(inds)=1:P;
    rn_w=rn_w/sum(rn_w);
    [val best_ind]=max(rn_w);
    %best_ind
    bb(i)=w(best_ind);
    ob(i)=mean(w);
    secilenler = randsample(P,P,true,rn_w);
    % yeni bireyleri �ret % tek/cift noktali crossover
    YB=zeros(P,u); % yeni bireyler
    for j=1:P/2
        b1=B(secilenler(j),:);
        b2=B(secilenler(j+(P/2)),:);
        if cross==1 % tek noktal� crossover
            kesme=round((u-3)*rand(1,1))+2; % 2 - (u-1) arasi sayi
            YB(j,:)      =[b1(1:kesme) b2(kesme+1:end)];
            YB(j+(P/2),:)=[b2(1:kesme) b1(kesme+1:end)];
        else
            % cift noktal� crossover
            kesme=round((u-3)*rand(1,2))+2; % 2 - (u-1) arasi 2 sayi
            kesme=sort(kesme); % kucukten buyuNesil_Sayisie sirala
            YB(j,:)      =[b1(1:kesme(1)) b2(kesme(1)+1:kesme(2)) b1(kesme(2)+1:end) ];
            YB(j+(P/2),:)=[b2(1:kesme(1)) b1(kesme(1)+1:kesme(2)) b2(kesme(2)+1:end) ];
        end
    end
    if BK>0 % B deki en iyi BK deNesil_Sayisieri YB ye kopyala
        YB(inds(BK+1:end),:)=B(inds(BK+1:end),:); 
    end
    % mutasyon uyNesil_Sayisiula
    d_ind=rand(P,u)<mu; % deNesil_Sayisiisecek hucreler
    yy=round(8*rand(P,u)); % nelerle deNesil_Sayisiisecekleri
    YB(d_ind)=yy(d_ind);
    B=YB; % yeni nesil hazir
    
end

plot(bb); % en iyileri ciz
hold on;
plot(ob); % ortlamay� �iz

% en iyi bireyi ciz
M=zeros(Mx,My); % ortam� 0 la
birey=B(best_ind,:);
% bireyin hareketini olustur
kxy=bas;
M(kxy(1),kxy(2))=1;  % baslanNesil_Sayisi��ta
for k=1:u
    kxy=kxy+hareket(birey(k)+1,:); % hareket et
    M(kxy(1),kxy(2))=k+1;
end
M(son(1),son(2))=64;
% path olustu
fiNesil_Sayisiure;imaNesil_Sayisie((50/u)*M);

