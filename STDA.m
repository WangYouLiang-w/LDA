function [Ac_STDA,code_dvtest_stda,code_latest_stda]=STDA_xx0(Train_trials,Train_label,Test_trials,Test_label,t)
%% STDA
fprintf('Now Algorithm is STDA......\n');

Train_trials_stda=reshape(Train_trials,size(Train_trials,1),[],t);
Test_trials_stda=reshape(Test_trials,size(Test_trials,1),[],t);
% Train_trials=permute(Train_trials,[1,3,2]);
for L=5%:size(Train_trials_stda,2)
    W2=eye(size(Train_trials_stda,3));
    for n=1:size(Train_trials_stda,1)
        Y{1}(:,:,n)=squeeze(Train_trials_stda(n,:,:))*W2;
    end
    TrainNum=size(Train_trials_stda,1)/2;
    mY{1,1}=mean(Y{1}(:,:,1:TrainNum),3);
    mY{1,2}=mean(Y{1}(:,:,TrainNum+1:end),3);
    MY{1}=mean(Y{1},3);

    SB{1}=TrainNum*(mY{1,1}-MY{1})*(mY{1,1}-MY{1})'+TrainNum*(mY{1,2}-MY{1})*(mY{1,2}-MY{1})';
    SW{1}=mean((Y{1}(:,:,1:TrainNum)-repmat(mY{1,1},1,1,TrainNum)),3)*mean((Y{1}(:,:,1:TrainNum)-repmat(mY{1,1},1,1,TrainNum)),3)'...
               +mean((Y{1}(:,:,1+TrainNum:end)-repmat(mY{1,2},1,1,TrainNum)),3)*mean((Y{1}(:,:,1+TrainNum:end)-repmat(mY{1,2},1,1,TrainNum)),3)';
    [W1,D1]=eig((inv(SW{1})*SB{1}));
    W1=W1(:,1:L);
    for n=1:size(Train_trials_stda,1)
        Y{2}(:,:,n)=(W1'*squeeze(Train_trials_stda(n,:,:)))';
    end

    mY{2,1}=mean(Y{2}(:,:,1:TrainNum),3);
    mY{2,2}=mean(Y{2}(:,:,TrainNum+1:end),3);
    MY{2}=mean(Y{2},3);

    SB{2}=TrainNum*(mY{2,1}-MY{2})*(mY{2,1}-MY{2})'+TrainNum*(mY{2,2}-MY{2})*(mY{2,2}-MY{2})';

    Ytemp1=Y{2}(:,:,1:TrainNum)-repmat(mY{2,1},1,1,TrainNum);
    Ytemp2=Y{2}(:,:,1+TrainNum:end)-repmat(mY{2,2},1,1,TrainNum);
    for n=1:size(Train_trials_stda,1)/2
        Yttemp1(:,:,n)=Ytemp1(:,:,n)*Ytemp1(:,:,n)';
        Yttemp2(:,:,n)=Ytemp2(:,:,n)*Ytemp2(:,:,n)';
    end
    SW{2}=sum(Yttemp1,3)+sum(Yttemp2,3); 
    [W2,D2]=eig(((pinv(SW{2})*SB{2})+(pinv(SW{2})*SB{2})')/2);
    W2=W2(:,1:L);

    for n=1:size(Train_trials_stda,1)
        fTrain_trials(:,n)=reshape(W1'*squeeze(Train_trials_stda(n,:,:))*W2,[],1);
    end
    mu1=mean(fTrain_trials(:,1:TrainNum),2);
    mu2=mean(fTrain_trials(:,1+TrainNum:end),2);
    sigema1=cov(fTrain_trials(:,1:TrainNum)');
    sigema2=cov(fTrain_trials(:,1+TrainNum:end)');
    sigema=(sigema1+sigema2)/2;
    wf=pinv(sigema)*(mu2-mu1);
    
%     Test_trials=reshape(Test_trials,size(Test_trials,2),[],32);
%     Test_trials=permute(Test_trials,[1,3,2]);
    for n=1:size(Test_trials_stda,1)
        f=reshape(W1'*squeeze(Test_trials_stda(n,:,:))*W2,[],1);
        H=wf'*f;
        code_STDA(L-1,n)=H;
        if H>0
            meanclass(n)=1;
        else
            meanclass(n)=-1;
        end
    end
    count=sum(meanclass'==Test_label);
    Acc_STDA(L-1)=sum(count)./size(Test_trials_stda,1);
    
clear W1 W2 Y* MY m* sigema* wf SB SW fTrain_trials meanclass
end 
m=mean(Acc_STDA,3);
[a,b]=max(m);

Ac_STDA=squeeze(Acc_STDA(:,b));
code_dvtest_stda=squeeze(code_STDA(b,:));
code_latest_stda=[zeros(size(Test_trials,1)/2,1)-1;ones(size(Test_trials,1)/2,1)];